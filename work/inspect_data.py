import time
import importlib.util
import pandas as pd
import numpy as np
import psutil

print('memory', psutil.virtual_memory(), flush=True)
print('modules', {m: importlib.util.find_spec(m) is not None for m in ['catboost', 'torch', 'nbformat', 'nbclient', 'nltk']}, flush=True)
cols = ['search_query', 'search_location_id', 'search_category', 'search_is_delivery_search', 'search_infm_params_text', 'item_id', 'item_category_id', 'item_location_id']
t = pd.read_parquet('dataset/train.parquet', columns=cols)
i = pd.read_parquet('dataset/benchmark_items.parquet', columns=['item_id', 'item_category_id', 'item_location_id'])
q = pd.read_parquet('dataset/benchmark_queries.parquet')
print('shapes', t.shape, i.shape, q.shape)
print('item coverage', t.item_id.isin(i.item_id).mean(), t.item_id.nunique(), i.item_id.nunique())
print('category match', t.search_category.eq(t.item_category_id).mean())
print('location match', t.search_location_id.eq(t.item_location_id).mean())
print('train cats', t.search_category.value_counts().head(10).to_dict())
print('test cats', q.search_category.value_counts().head(10).to_dict())
print('query text overlap', q.search_query.isin(t.search_query).mean())
print('query locations', q.search_location_id.value_counts().head(10).to_dict())
print('train unique text', t.search_query.nunique())
print('matching category coverage', t.loc[t.item_id.isin(i.item_id)].search_category.eq(t.loc[t.item_id.isin(i.item_id)].item_category_id).mean())
print('eval_sample', pd.read_parquet('eval_sample.parquet').shape)
print(pd.read_parquet('eval_sample.parquet').head(2).to_string())
print('delivery', t.search_is_delivery_search.value_counts().to_dict())
