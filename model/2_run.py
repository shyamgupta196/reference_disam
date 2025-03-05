# %%
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from logger_and_config import logger, config
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import util
from sklearn.metrics import average_precision_score
import time
from time import mktime
import logging
import torch
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm

# %%
logging.basicConfig(level=logging.INFO)
logger.info('PROCESS START: SIMILARITY P1 MAIN')

machine = 'GPU-SERVER'
if ('c:\\Users\\local\\' in __file__ or '[SOMEONES LOCAL PATH HEADER]' in __file__):
    machine = 'LOCAL'
    logger.info('running on LOCAL machine...')
else:
    logger.info('running on server...')


# %%
preproc, scifilter = config['preproc'], config['scifilter']
logger.info(f'--- {preproc=} --- {scifilter=}')

default_dataset = config['default_dataset']
logger.info(f'--- {default_dataset=}')
if machine == 'GPU-SERVER':
    p_collection, p_query = Path(config[default_dataset]['files'][0]),  Path(config[default_dataset]['files'][1])
    ps_feature = config['dirs']['feature']
elif machine == 'LOCAL':
    p_collection, p_query = Path(config[default_dataset]['files'][0].replace('/data_ssds/disk04/kartalym/reference_disam','..')),  Path(config[default_dataset]['files'][1].replace('/data_ssds/disk04/kartalym/reference_disam','..'))
    ps_feature = config['dirs']['feature'].replace('/data_ssds/disk11/kartalym/reference_disam','..')

MODEL_TEXT_NAME, MODEL_TEXTNE_NAME, MODEL_TITLE_NAME = config['models']['sentence_transformers'][config['default_models']['text']], config['models']['sentence_transformers']['MiniLM-L6'], config['models']['sentence_transformers'][config['default_models']['title']] 
logger.info(f'--- {MODEL_TEXT_NAME=} --- {MODEL_TEXTNE_NAME=} --- {MODEL_TITLE_NAME=}')

TEXT_COL, PREPROC_TEXT_COL, TEXTNE_COL, TITLE_COL, NEWS_TITLE_COL, LABEL_COL = config['columns']['text'], config['columns']['preproc_text'], config['columns']['text_named_entities'], config['columns']['title'], config['columns']['news_title'], config['columns']['label']
logger.info(f'--- {TEXT_COL=} --- {PREPROC_TEXT_COL=} --- {TEXTNE_COL=} --- {TITLE_COL=} --- {NEWS_TITLE_COL=} --- {LABEL_COL=}')

min_pri_mention, l_min_pri_mention = config['min_pri_mention'], config['l_min_pri_mention']
logger.info(f'--- {min_pri_mention=} --- {l_min_pri_mention=}')
SCORE_FUNC, L_K = config['score_func'], config['eval_k']
logger.info(f'--- {SCORE_FUNC=} --- {L_K=}')

is_time_window, l_time_interval = config['time_window'], config['l_time_interval']
logger.info(f'--- {is_time_window=} --- {l_time_interval=}')

l_features = config['l_features']
type_score_merge = config['type_score_merge']
dedup_col_sample = True
if len(l_features) > 1:
    dedup_col_sample = False
logger.info(f'---{l_features=} --- {dedup_col_sample=} --- {type_score_merge=}')

select_n, take_avg = config['select_n'], config['take_avg']
logger.info(f'--- {select_n=} --- {take_avg=}')

# %%

df_collection_all, df_query_all = pd.read_pickle(p_collection), pd.read_pickle(p_query)

df_collection_all.reset_index(inplace=True)
df_query_all.reset_index(inplace=True)

logger.info(f'read files - collection: {p_collection.name} ({str(df_collection_all.shape)}) and query: {p_query.name} ({str(df_query_all.shape)})')

TS_ONE_DAY = 86400

d_tf = {
    1: '%a %b %d %H:%M:%S +0000 %Y',
    2: '%Y-%m-%dT%H:%M:%S.%fZ',
    3: '%Y-%m-%d',
    4: '%a, %d %b %Y %H:%M:%S GMT',
    5: '%Y-%m-%d %H:%M:%S+00:00'
}

time_format_q = 5

# %%    EVAL FUNCTIONS


def euc_dist(npa_1, npa_2):
    """returns euclidian distance between two arrays

    Args:
        npa_1 (_type_): 2d np array
        npa_2 (_type_): 2d np array

    Returns:
        _type_: _description_
    """
    l_1 = []
    for i in npa_1:
        l_2 = []
        for j in npa_2:
            l_2.append(np.linalg.norm(i-j))
        l_1.append(l_2)
    return torch.tensor(np.array(l_1))


def score_func(vec_1, vec_2, type=SCORE_FUNC):
    if type == 'cos':
        return util.cos_sim(vec_1, vec_2)
    elif type == 'euc':
        if len(vec_1.shape) == 1:
            vec_1 = [vec_1]
        if len(vec_2.shape) == 1:
            vec_2 = [vec_2]
        return euc_dist(vec_1, vec_2)


def eval_avp(l_true, l_score):
    if any(l_true):
        return average_precision_score(np.array(l_true), np.array(l_score))
    else:
        return 0


def rrank(golds, preds):
    l1 = golds
    l2 = preds
    lz = zip(l1, l2)
    res = sorted(lz, key=lambda x: x[1], reverse=True)
    r = [a for (a, b) in res].index(1)
    return 1/(r+1)


def evaluate(l_collection_labels, query_label, score, k, type=SCORE_FUNC):
    # logger.info(f'evaluation started for map@{k} and acc@{k}')
    avp = 0
    acc = 0
    rr = 0
    if type == 'cos':
        l_topind = np.argsort(score)[-k:]
    elif type == 'euc':
        l_topind = np.argsort(score)[:k]
    l_topval = [score[i] for i in l_topind]
    goldlbl = query_label
    l_predlbl = [l_collection_labels[i] for i in l_topind]
    match = []
    for a in l_predlbl:
        if a == goldlbl:  # or l_topval[l_predlbl.index(a)]>0.90:
            match.append(1)
        else:
            match.append(0)

    if any(match):
        avp = (eval_avp(match, l_topval))
        acc = (1)
        rr = rrank(match, l_topval)

    return (avp, acc, rr)


# %%    LOAD EMBEDDINGS
d_f_npa = {}


def load_embeddings(type: str, model_name: str, column: str):
    if type == 'col':
        type_df = 'npa_col_' + column
        size_df = len(df_collection_all)
        p_df = p_collection
    elif type == 'query':
        type_df = 'npa_query_' + column
        size_df = len(df_query_all)
        p_df = p_query
    else:
        logger.info('Please state the type of dataset')

    fp_c = Path(ps_feature + p_df.stem + '-' + model_name.replace('/', '_').replace('-', '_') + '-' + column + '.npy')
    d_f_npa[type_df] = np.load(fp_c, mmap_mode='c')

    if len(d_f_npa[type_df]) == size_df:
        logger.info(f' {column} embedding vector array sizes are aligned with dataset sizes.')
    else:
        logger.error(f' {column} embedding vector array sizes are NOT aligned with dataset sizes: {len(d_f_npa[type_df])=}, {size_df=}')


# %%
df_collection = pd.DataFrame()

d_results = {}
for k in L_K:
    d_results['l_acc@' + str(k)] = []
    d_results['l_map@' + str(k)] = []
    d_results['l_rr@' + str(k)] = []


def run(run_df_query, run_df_col, take_avr):
    df_collection = run_df_col.copy()
    for k in L_K:
        d_results['l_acc@' + str(k)] = []
        d_results['l_map@' + str(k)] = []
        d_results['l_rr@' + str(k)] = []
    try:

        for i, row in run_df_query[:].iterrows():

            ts = mktime(time.strptime(str(row['tweet_created_at']), d_tf[time_format_q]))
            df_sample = df_collection[(df_collection['time'] < ts)]
            # df_sample = df_collection.copy()

            l_collection_labels = []
            l_scores = []

            if 'text_text_sim' in l_features:

                l_selected_ind_time = df_sample.index.to_list()
                l_collection_labels = df_collection.loc[l_selected_ind_time][LABEL_COL].to_list()
                sdf = np.take(d_f_npa['npa_col_text'], l_selected_ind_time, axis=0)
                npa_q_sentemb = np.take(d_f_npa['npa_query_tweet_text'], [i], axis=0)[0]
                l_scores.append(score_func(npa_q_sentemb, sdf).tolist()[0])

            if 'text_title_sim' in l_features:
                l_selected_ind_time = df_sample.index.to_list()
                l_collection_labels = df_collection.loc[l_selected_ind_time][LABEL_COL].to_list()
                sdf = np.take(d_f_npa['npa_col_title'], l_selected_ind_time, axis=0)
                npa_q_sentemb = np.take(d_f_npa['npa_query_text'], [i], axis=0)[0]
                l_scores.append(util.cos_sim(npa_q_sentemb, sdf).tolist()[0])

            if i % 20000 == 0:
                logger.debug(f'cnt {i}')
                for a in l_scores:
                    logger.info(a[:5])

            score = 0

            if 'sum' == type_score_merge:
                score = np.sum(l_scores, axis=0)
            elif 'max' == type_score_merge:
                npa_scores = np.array(l_scores)
                score = npa_scores.max(axis=0)

            else:
                score = np.sum(l_scores, axis=0)

            if i % 250 == 0:
                with open("d_results.json", "w") as outfile:
                    json.dump(d_results, outfile)
                logger.debug(f'cnt {i}')
                logger.info(score[:5])
                for k, v in d_results.items():
                    logger.info(f'{k}: { np.mean(v)}')
            for k in L_K:
                # logger.info(l_selected_ind_time)
                (avp, acc, rr) = evaluate(l_collection_labels, row[LABEL_COL], score, k)
                l_temp = d_results.get('l_map@' + str(k), [])
                l_temp.append(avp)
                d_results['l_map@' + str(k)] = l_temp
                l_temp = d_results.get('l_acc@' + str(k), [])
                l_temp.append(acc)
                d_results['l_acc@' + str(k)] = l_temp
                l_temp = d_results.get('l_rr@' + str(k), [])
                l_temp.append(rr)
                d_results['l_rr@' + str(k)] = l_temp

        logger.info('cosine similarities computed without time_interval')
        for k, v in d_results.items():
            logger.info(f'{k}: { np.mean(v)}')

        return d_results

    except Exception as err:
        logger.error(err)


# %% CONFIGURE DATASETS

if 'text_text_sim' in l_features:
    load_embeddings('col', MODEL_TEXT_NAME.replace('/', '_').replace('-', '_'), TEXT_COL)
    load_embeddings('query', MODEL_TEXT_NAME.replace('/', '_').replace('-', '_'), 'tweet_text')
if 'text_title_sim' in l_features:
    load_embeddings('col', MODEL_TITLE_NAME.replace('/', '_').replace('-', '_'), TITLE_COL)
    load_embeddings('query', MODEL_TEXT_NAME.replace('/', '_').replace('-', '_'), TEXT_COL)

# %%

logger.info(f'Size of col df_query_all: {df_query_all.shape} and number of sciworks: {len(set(df_query_all.label))}')

# %% MULTIPROCESSING


def mp_func(data, run_df_col, func, workers=4, splitted=False, concat=True):

    if not splitted:
        logger.debug("split data")
        data = np.array_split(data, workers)

    logger.debug("start multiprocessing")
    try:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(func, split, run_df_col) for split in data]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
                res = future.result()
                results.append(res)

        if concat:
            logger.debug("concatenate results")
            super_dict = defaultdict(set)  # uses set to avoid duplicates
            for d in results:
                for k, v in d.items():  # use d.iteritems() in python 2
                    l_temp = super_dict.get(k, [])
                    l_temp.extend(v)
                    super_dict[k] = l_temp
            results = super_dict
    except KeyboardInterrupt:
        executor.terminate()
    logger.debug("finished")

    return results


def function_to_run(data, run_df_col):
    res = run(data, run_df_col, take_avg)
    return res


# %% MAIN RUN

for k in L_K:
    d_results['l_acc@' + str(k)] = []
    d_results['l_map@' + str(k)] = []
    d_results['l_rr@' + str(k)] = []

df_collection = df_collection_all.copy()

df_collection['time'] = df_collection.time.values.astype(np.int64) // 10 ** 9
df_collection.sort_values(by='time', inplace=True)

df_query = df_query_all.copy()[:]

results = mp_func(df_query, df_collection, function_to_run, 10, False, True)
with open("results_multi.json", "w") as outfile:
    json.dump(results, outfile)
for k, v in results.items():
    logger.info(f'{k}: { np.mean(v)}')
