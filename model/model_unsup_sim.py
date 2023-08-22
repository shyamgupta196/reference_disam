#%%
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import json
from sklearn.metrics import average_precision_score
from logger_and_config import logger, config
# %%
logger.info('PROCESS START: UNSUP_SIMILARITY')

#%% CONFIG

default_dataset = config['default_dataset']

p_collection= Path(config[default_dataset]['files'][0])
p_query= Path(config[default_dataset]['files'][1])

MODEL_NAME = 'sentence-transformers/sentence-t5-base'
TEXT_COL = 'text'
LABEL_COL = 'label'
L_K = [1,5,10]


# %%    READ FILES
df_collection = pd.read_pickle(p_collection)
df_query = pd.read_pickle(p_query)

logger.info(f'read files - collection: {p_collection.name} and query: {p_collection.name}')

# %%    FUNCTIONS
def convert_text2sentemd(model_name, df_data, text_col: str):

    model = SentenceTransformer(model_name)
    result = model.encode(df_data[text_col].to_list())
    
    return result

# %%    EVAL FUNCTIONS
def eval_avp(l_true, l_score):
    if any(l_true) == True:
        return average_precision_score(np.array(l_true), np.array(l_score))
    else:
        return 0
    
def evaluate(l_collection_labels, l_query_labels, scores, k):
    logger.info(f'evaluation started for map@{k} and acc@{k}')
    l_avp = []
    l_acc = []
    for idx, cs in enumerate(scores):
        l_topind = np.argsort(cs)[-k:]
        l_topval = [cs[i] for i in l_topind]

        
        goldlbl = l_query_labels[idx]
        #goldlbl = df_query[LABEL_COL].to_list()[idx]
        #predlbl = df_collection[LABEL_COL].to_list()

        l_predlbl = [l_collection_labels[i] for i in l_topind]
        match = []
        for a in l_predlbl:
            #print(a, trlbl)
            if a == goldlbl:
                match.append(1)
            else:
                match.append(0)

        if any(match) == True:
            l_avp.append(eval_avp(match, l_topval))
            l_acc.append(1)
        else:
            l_avp.append(0)
            l_acc.append(0)

        map = np.mean(l_avp)
        acc = np.mean(l_acc)
    logger.info(f'map at {k}: {map}')
    logger.info(f'acc at {k}: {acc}')
# %%

try:
    npa_col_sentemb = convert_text2sentemd(MODEL_NAME, df_collection, TEXT_COL)
    logger.info(f'sent_emd - {str(npa_col_sentemb.shape)} with {MODEL_NAME} for {p_collection.name} completed')
    
    model = SentenceTransformer(MODEL_NAME)
    scores = []
    for i, row in df_query.iterrows():
        if i%10000==0:
            logger.debug(f'cnt {i}')
        npa_q_sentemb = model.encode(row[TEXT_COL])
        cs = util.cos_sim(npa_q_sentemb, npa_col_sentemb)
        scores.append(cs.tolist()[0])

    #np.save('sim_scores.pkl', np.array(scores))  #uncomment this if want to save scores array
    logger.info(f'cosine similarities computed')

    for k in L_K:
        evaluate(df_collection[LABEL_COL].to_list(), df_query[LABEL_COL].to_list(), scores, k)




except Exception as err:
    logger.error(err)

logger.info('PROCESS FINISH: UNSUP_SIMILARITY')

