# %%
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
# %%    CONFIG


SUFFIX = ''

FILE_TRAIN =   '../data/train__01_min_5.tsv'
FILE_TEST =    '../data/test__01_min_5.tsv'


L_TRAIN_COLS = ['ln', 'id', 'text', 'c_text', 'class']
L_TEST_COLS = ['ln', 'id', 'text', 'c_text', 'class']

LABEL_COLUMN = 'class'     # label column
INSTANCE_COLUMN = 'c_text' # text without USERMENTION and URL

SEP = '\t'

L_MODEL =     ['sentence-transformers/sentence-t5-base',
               'all-mpnet-base-v2', 
               'allenai/scibert_scivocab_uncased', 
               'bert-base-uncased', 
               'sentence-transformers/all-MiniLM-L6-v2', 
               'sentence-transformers/allenai-specter', 
               'vinai/bertweet-base']

BATCH_SIZE = 16
SCORE_FUNC = 'cos_sim'
# %%    GLOB VARS
df_train = pd.DataFrame()
df_test = pd.DataFrame()
d_model_results = {}


# %%

### DATA FUNCTIONS ###


def read_data(f_train, f_test):

    df_train = pd.read_csv(f_train, sep= SEP, names= L_TRAIN_COLS, header=0)
    df_test = pd.read_csv(f_test, sep= SEP, names= L_TEST_COLS, header=0)

    print(f'{f_train} and {f_test} are loaded in df_train and df_test')
    return df_train, df_test


def create_corpus(df_train):

    d_train_url_id = {k: g['id'].tolist() for k,g in df_train.groupby('class')}

    corpus = {}

    for k, v in d_train_url_id.items():

        for i, id in enumerate(v) :

            text = str(df_train[df_train['id']==id][INSTANCE_COLUMN].values[0])
            dict_text = {}
            dict_text['title'] = str(id)
            dict_text['text'] = text

            corpus[str(id) + '_' + str(i)] = dict_text


    print(f'CORPUS from df_train is created with size of {len(corpus)}')
    return corpus


def create_queries(df_test):

    l_test_ids = df_test['id'].to_list()

    queries = {}

    for i, id in enumerate(l_test_ids):

        text = str(df_test[df_test['id']==id]['c_text'].values[0])
        queries[id] = text 
    
    print(f'QUERIES from df_test is created with size of {len(queries)}')
    return queries


### MODEL FUNCTIONS ###


def load_model(model_name, k: int):
    """_summary_

    Args:
        model_name (_type_): _description_
        k (int): Number of results to retrieve, fetching top k results

    Returns:
        _type_: _description_
    """
    
    k_values = [k-1]
    pre_model = DRES(models.SentenceBERT(model_name), BATCH_SIZE=BATCH_SIZE)
    model = EvaluateRetrieval(pre_model, SCORE_FUNC=SCORE_FUNC, k_values=k_values) 
    
    return model


def retrieve(model, corpus, queries):

    results = model.retrieve(corpus, queries)

    print(f'Results are retrieved from')
    return results


### EVAL FUNCTIONS ###


def eval_avp(l_true, l_score):
    if any(l_true) == True:
        return average_precision_score(np.array(l_true), np.array(l_score))
    else:
        return 0

 
def evaluate(df_train, df_test, d_results, l_eval_k):
    """Evaluate results in terms of Mean Avg Precision and Top N Accuracy

    Args:
        df_train (_type_): _description_
        df_test (_type_): _description_
        d_results (_type_): results to evaluate
        l_eval_k (_type_): metric parameter list such as MAP@k, MUST be <= than retrieved result count
    """

    dict_train_id_url = dict(zip(df_train['id'].to_list(), df_train['class'].to_list()))
    dict_test_id_url = dict(zip(df_test['id'].to_list(), df_test['class'].to_list()))

    for eval_k in l_eval_k:
        
        l_avp = []
        l_acc = []
        for s_id_q, match in d_results.items():

            id_q=int(s_id_q)
            y_true = []
            y_scores = []
            ord_match = dict(sorted(match.items(), key=lambda item: item[1], reverse=True)[:eval_k])
            for k, v in ord_match.items():
                id = k.split('_')[0]
                if dict_train_id_url[int(id)] == dict_test_id_url[id_q]:
                    y_true.append(1)
                else:
                    y_true.append(0)
                y_scores.append(v)

            if any(y_true) == True:
                l_avp.append(eval_avp(y_true, y_scores))
                l_acc.append(1)
            else:
                l_avp.append(0)
                l_acc.append(0)

        map = np.mean(l_avp)
        acc = np.mean(l_acc)
        print(f'MAP@{eval_k}: {map}, TOP-{eval_k} ACC: {acc}')

    


#%%
df_train, df_test = read_data(FILE_TRAIN, FILE_TEST)

#%%
corpus = create_corpus(df_train)
queries = create_queries(df_test)

# %%
model = load_model(L_MODEL[0], 10)

# %%
results = retrieve(model, corpus, queries)

# %%
d_model_results[L_MODEL[0]] = results

# %%
evaluate(df_train, df_test, results, [1,5,10])

# %%
