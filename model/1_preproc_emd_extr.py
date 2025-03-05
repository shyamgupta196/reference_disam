# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from logger_and_config import logger, config
import torch
# %%
logger.info('PROCESS START: SENT_EMB EXTRACTION')
# %%

l_features = config['l_features']
logger.info(f'--- {l_features=}')

default_dataset = config['default_dataset']
logger.info(f'--- {default_dataset=}')
preproc = config['preproc']
logger.info(f'--- {preproc=}')
scifilter = config['scifilter']
logger.info(f'--- {scifilter=}')

p_collection =  Path(config[default_dataset]['files'][0]) # Path('../test/cord_col_7718.pkl')
p_query =  Path(config[default_dataset]['files'][1]) # Path('../test/cord_q_252_7718.pkl')

MODEL_TEXT_NAME = config['models']['sentence_transformers'][config['default_models']['text']] # 'sentence-transformers/sentence-t5-base'
logger.info(f'--- {MODEL_TEXT_NAME=}')
MODEL_TEXTNE_NAME = config['models']['sentence_transformers']['MiniLM-L6'] # 'sentence-transformers/all-MiniLM-L6-v2'
logger.info(f'--- {MODEL_TEXTNE_NAME=}')
MODEL_TITLE_NAME = config['models']['sentence_transformers'][config['default_models']['title']] # 'sentence-transformers/sentence-t5-base'
logger.info(f'--- {MODEL_TITLE_NAME=}')
TEXT_COL = config['columns']['text'] #'text'
logger.info(f'--- {TEXT_COL=}')
PREPROC_TEXT_COL = config['columns']['preproc_text'] #'tweet_text'
logger.info(f'--- {PREPROC_TEXT_COL=}')
TEXTNE_COL = config['columns']['text_named_entities'] #'ner'
logger.info(f'--- {TEXTNE_COL=}')
TITLE_COL = config['columns']['title'] #'title'
logger.info(f'--- {TITLE_COL=}')
NEWS_TITLE_COL = config['columns']['news_title'] #'news_title'
logger.info(f'--- {NEWS_TITLE_COL=}')

device = torch.device('cuda:0')
# %%    READ FILES

df_collection = pd.read_pickle(p_collection)
df_query = pd.read_pickle(p_query)

logger.info(f'read files - collection: {p_collection.name} ({str(df_collection.shape)}) and query: {p_query.name} ({str(df_query.shape)})')

# %%


def ftr_sent_embed(model_name, df_data, text_col: str, fp):

    fp = Path(config['dirs']['feature'] + fp.stem + '-' + model_name.replace('/', '_').replace('-', '_') + '-' + text_col + '.npy')
    if fp.exists():
        logger.info(f'{fp.name} already exists.')
        return fp
    else:
        model = SentenceTransformer(model_name)
        model.to(device)
        print(df_data[text_col][:10])
        result = model.encode(df_data[text_col].to_list())
        np.save(fp, result)
        del result
        del model
        logger.info(f'{fp.name} sent_embed completed and saved.')
        return fp


# %%

try:
    d_model = {}
    d_coll_sentemb = {}
    if ('text' in l_features):
        ftr_sent_embed(MODEL_TEXT_NAME, df_collection, TEXT_COL, p_collection)
        ftr_sent_embed(MODEL_TEXT_NAME, df_query, TEXT_COL, p_query)
    if ('title' in l_features):
        df_collection = df_collection.astype({'title': str})
        df_collection['text'] = df_collection.title
        ftr_sent_embed(MODEL_TEXT_NAME, df_collection, 'text', p_collection)
        df_query = df_query.astype({'tweet_text': str})
        ftr_sent_embed(MODEL_TEXT_NAME, df_query, 'tweet_text', p_query)
    if ('ne' in l_features):
        ftr_sent_embed(MODEL_TEXTNE_NAME, df_collection, TEXTNE_COL, p_collection)
        ftr_sent_embed(MODEL_TEXTNE_NAME, df_query, TEXTNE_COL, p_query)
except:
    raise

# %%
