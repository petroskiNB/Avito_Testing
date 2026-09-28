# %% markdown
## Тестовое задание Avito

Кандидатогенератор без нейросетевого обучения: BM25 по словам, символьный TF-IDF по заголовкам и мягкий учет географии. Индексы строятся один раз, поиск работает по разреженным матрицам.

# %% code
# импорт библиотек и считывание данных + 5 строчек из train
import os
import re
import gc
import html
import json
import time
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import RussianStemmer
from tqdm.auto import tqdm
import joblib
from catboost import CatBoostClassifier

START_TIME = time.time()
RANDOM_STATE = 42
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
Path('work').mkdir(exist_ok=True)

# для train не загружаем длинные описания: они не нужны для этого решения
query_cols = ['search_query', 'search_location_id', 'search_is_delivery_search',
              'search_infm_params_text', 'search_category']
train = pd.read_parquet('dataset/train.parquet', columns=query_cols + [
    'item_id', 'item_microcat_id', 'item_location_id'])
queries = pd.read_parquet('dataset/benchmark_queries.parquet')
items = pd.read_parquet('dataset/benchmark_items.parquet').drop_duplicates('item_id').reset_index(drop=True)
print('Размеры train, queries, items:', train.shape, queries.shape, items.shape)
print(train.head().to_string(index=False))

# %% code
# проверяем пропуски в значениях, а не в названиях колонок
print('Пропуски в запросах:')
print(queries.isna().sum())
print('Дубликаты query_id:', queries['query_id'].duplicated().sum())
print('Дубликаты item_id:', items['item_id'].duplicated().sum())

item_ids = items['item_id'].to_numpy()
item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
available = train['item_id'].isin(item_to_idx)
print(f'Доля train-пар, объявления которых есть в корпусе: {available.mean():.4%}')
print('Старый Recall на всех train-парах штрафует за объявления, которых нет в корпусе.')

# %% markdown
#### **Сделаем преобразование данных перед построением кандидатогенератора**

Сохраняем цифры и латиницу, заменяем ё на е. Слова приводим к основам русским SnowballStemmer. Заголовок и описание индексируем отдельно, чтобы длинное описание не заглушало заголовок. Фильтры не добавляем в основной запрос: общие слова «вид услуги» иначе вытесняют его смысл.

# %% code
stemmer = RussianStemmer()
stop_words = set('и в во на с со к ко от до по из за у о об для при а но или это как что не без под над вид услуги тип место оказания рейтинг пользователя звезды выше онлайн запись'.split())

def clean_text_basic(text):
    if pd.isna(text):
        return ''
    text = html.unescape(str(text)).lower().replace('ё', 'е')
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
    return ' '.join(text.split())

@lru_cache(maxsize=400000)
def stem_word(word):
    return stemmer.stem(word) if re.search('[а-я]', word) else word

def prepare_word_text(text):
    return ' '.join(stem_word(word) for word in clean_text_basic(text).split()
                    if word not in stop_words)

for col in ['item_title_raw', 'item_description_raw', 'item_infm_params_text']:
    items[col] = items[col].fillna('')

items['text_title'] = items['item_title_raw'].map(prepare_word_text)
# ограничение защищает от очень длинных SEO-описаний
items['text_description'] = items['item_description_raw'].str.slice(0, 8000).map(prepare_word_text)
items['text_params'] = items['item_infm_params_text'].map(prepare_word_text)
items['text_char'] = items['item_title_raw'].map(clean_text_basic)
train['query_norm'] = train['search_query'].map(clean_text_basic)
queries['query_norm'] = queries['search_query'].map(clean_text_basic)

# ключ включает также доставку: это часть контекста поиска
train['query_key'] = train[query_cols].fillna('').astype(str).agg('|'.join, axis=1)
train_grouped = train.groupby('query_key', sort=False).agg({
    **{col: 'first' for col in query_cols},
    'query_norm': 'first',
    'item_id': lambda x: list(set(x))
}).reset_index()
train_grouped['relevant_all'] = train_grouped['item_id'].map(len)
train_grouped['item_id'] = train_grouped['item_id'].map(lambda ids: [x for x in ids if x in item_to_idx])
train_grouped['relevant_available'] = train_grouped['item_id'].map(len)
coverage_ceiling = (train_grouped['relevant_available'] / train_grouped['relevant_all']).mean()
print(f'Потолок старого Recall на полном train: {coverage_ceiling:.4f}')

# фиксированный набор для подбора и отдельный holdout; одинаковые тексты не пересекаются
eligible = train_grouped[train_grouped['relevant_available'] > 0].copy()
unique_texts = eligible['query_norm'].drop_duplicates().sample(frac=1, random_state=RANDOM_STATE)
tune_texts = set(unique_texts.iloc[:700])
holdout_texts = set(unique_texts.iloc[700:1400])
tune_sample = eligible[eligible['query_norm'].isin(tune_texts)].sample(frac=1, random_state=RANDOM_STATE).drop_duplicates('query_norm').reset_index(drop=True)
eval_sample = eligible[eligible['query_norm'].isin(holdout_texts)].sample(frac=1, random_state=RANDOM_STATE).drop_duplicates('query_norm').reset_index(drop=True)
heldout_texts = tune_texts | holdout_texts
rank_train_texts = set(unique_texts.iloc[1400:3400])
rank_train_sample = eligible[eligible['query_norm'].isin(rank_train_texts)].sample(frac=1, random_state=RANDOM_STATE).drop_duplicates('query_norm').reset_index(drop=True)
# prior не видит также запросы, на которых учится ранкер: иначе его признаки были бы слишком хорошими
fit_train = train[~train['query_norm'].isin(heldout_texts | rank_train_texts)].copy()
print('Подбор / holdout / train:', len(tune_sample), len(eval_sample), len(fit_train))
print('Запросов для обучения ранкера:', len(rank_train_sample))
assert not set(fit_train['query_norm']) & heldout_texts
print('Подготовка текстов, секунд:', round(time.time() - START_TIME, 1))

# %% markdown
#### **Строим быстрые разреженные индексы**

BM25 учитывает частоту слова и длину документа. Символьный TF-IDF дополняет его для опечаток и разных написаний. Транспонированную матрицу сразу сохраняем в CSR: внутри поиска больше нет дорогой конвертации всего корпуса на каждом запросе.

# %% code
def build_bm25(texts, max_features=180000, k1=1.3, b=0.65):
    vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=max_features,
                          min_df=1, dtype=np.float32)
    mat = vec.fit_transform(texts).tocsr()
    lengths = np.asarray(mat.sum(axis=1)).ravel()
    df = np.bincount(mat.indices, minlength=mat.shape[1]).astype(np.float32)
    idf = np.log1p((mat.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1))
    mat.data *= (k1 + 1) / (mat.data + np.repeat(norm, np.diff(mat.indptr)))
    mat.data *= idf[mat.indices]
    return vec, mat.T.tocsr(), idf

word_vectorizers = {}
word_matrices = {}
word_idfs = {}
for name, texts in [('title', items['text_title']),
                    ('description', items['text_description']),
                    ('params', items['text_params'])]:
    t0 = time.time()
    vec, mat, idf = build_bm25(texts)
    word_vectorizers[name] = vec
    word_matrices[name] = mat
    word_idfs[name] = idf
    print(name, mat.shape, 'nnz:', mat.nnz, 'секунд:', round(time.time() - t0, 1), flush=True)

char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                max_features=100000, min_df=2,
                                sublinear_tf=True, dtype=np.float32)
char_matrix = char_vectorizer.fit_transform(items['text_char']).T.tocsr()
print('Char TF-IDF:', char_matrix.shape, 'nnz:', char_matrix.nnz, flush=True)

# %% code
# классификатор подкатегории: быстрый prior из train без текстов validation
def fit_priors(frame):
    grouped = frame.groupby(['query_norm', 'item_microcat_id'], sort=False).size().reset_index(name='weight')
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                          max_features=80000, sublinear_tf=True, dtype=np.float32)
    x = vec.fit_transform(grouped['query_norm'])
    model = MultinomialNB(alpha=0.05)
    model.fit(x, grouped['item_microcat_id'], sample_weight=np.sqrt(grouped['weight']))
    locations = frame.groupby(['search_location_id', 'item_location_id']).size()
    location_probs = {}
    for location, counts in locations.groupby(level=0):
        counts = counts.droplevel(0)
        location_probs[int(location)] = (counts / counts.sum()).to_dict()
    return vec, model, location_probs

prior_vectorizer, prior_model, location_probs = fit_priors(fit_train)
item_locations = items['item_location_id'].fillna(-1).to_numpy(dtype=np.int64)
item_categories = items['item_category_id'].fillna(-1).to_numpy(dtype=np.int64)
item_microcats = items['item_microcat_id'].fillna(-1).to_numpy(dtype=np.int64)
item_ratings = pd.to_numeric(items['item_rating'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
item_reviews = np.log1p(pd.to_numeric(items['item_rating_reviews_count'], errors='coerce').fillna(0).to_numpy(dtype=np.float32))
item_reviews /= max(item_reviews.max(), 1)
item_lats = pd.to_numeric(items['item_latitude'], errors='coerce').to_numpy(dtype=np.float32)
item_lons = pd.to_numeric(items['item_longitude'], errors='coerce').to_numpy(dtype=np.float32)
geo_frame = pd.DataFrame({'location': item_locations, 'lat': item_lats, 'lon': item_lons})
geo_centers = geo_frame.groupby('location')[['lat', 'lon']].median().to_dict('index')
microcat_columns = {int(value): idx for idx, value in enumerate(prior_model.classes_)}
print('Подкатегорий:', len(microcat_columns), 'локаций:', len(location_probs))

# %% markdown
#### **Поиск кандидатов и признаки для отбора топ-50**

Категория 0 означает поиск без ограничения категории. Географию учитываем мягко: кроме совпадения города используем частоты переходов между локациями в train и расстояние до центра города. Не отбрасываем удаленные услуги жестким фильтром.

# %% code
def bm25_scores(text, name):
    q = word_vectorizers[name].transform([text]).astype(np.float32)
    q.data[:] = 1
    scores = (q @ word_matrices[name]).toarray().ravel()
    # нормируем шкалу по весам слов запроса, а не по максимуму случайного объявления
    denom = word_idfs[name][q.indices].sum() if q.nnz else 1
    return scores / max(float(denom), 1)

def top_indices(scores, k=250):
    k = min(k, len(scores))
    idx = np.argpartition(scores, len(scores) - k)[-k:]
    return idx[np.argsort(-scores[idx], kind='stable')]

@lru_cache(maxsize=256)
def geo_scores(location):
    exact = (item_locations == location).astype(np.float32)
    probs = location_probs.get(location, {})
    unique_locs, inverse = np.unique(item_locations, return_inverse=True)
    p = np.array([probs.get(int(loc), 0) for loc in unique_locs], dtype=np.float32)[inverse]
    p = np.sqrt(p / max(float(p.max()), 1e-6))
    near = np.zeros(len(items), dtype=np.float32)
    if location in geo_centers:
        lat, lon = geo_centers[location]['lat'], geo_centers[location]['lon']
        dy = (item_lats - lat) * 111
        dx = (item_lons - lon) * 111 * np.cos(np.deg2rad(lat))
        distance = np.sqrt(dx * dx + dy * dy)
        near = np.nan_to_num(np.exp(-distance / 80), nan=0)
    return np.maximum(exact, np.maximum(0.85 * p, 0.65 * near))

def get_candidates(row, pool_size=250):
    text = prepare_word_text(row['search_query'])
    filters = prepare_word_text(row['search_infm_params_text'])
    title = bm25_scores(text, 'title')
    description = bm25_scores(text, 'description')
    params = bm25_scores(text, 'params')
    filter_score = bm25_scores(filters, 'params') if filters else np.zeros(len(items), dtype=np.float32)
    char = (char_vectorizer.transform([clean_text_basic(row['search_query'])]) @ char_matrix).toarray().ravel()
    geo = geo_scores(int(row['search_location_id']))
    if row.get('search_is_delivery_search', 0):
        geo = geo * 0.3
    prob = prior_model.predict_proba(prior_vectorizer.transform([clean_text_basic(row['search_query'])]))[0]
    micro = np.array([prob[microcat_columns[x]] if x in microcat_columns else 0
                      for x in np.unique(item_microcats)], dtype=np.float32)
    micro = micro[np.searchsorted(np.unique(item_microcats), item_microcats)]
    micro = np.sqrt(micro / max(float(micro.max()), 1e-6))
    category = int(row['search_category'])
    valid = np.ones(len(items), dtype=bool) if category == 0 else item_categories == category
    lexical = 0.5 * title + 0.35 * description + 0.15 * char
    # несколько каналов: глобальная лексика, локальный поиск, символы и фильтры
    channels = [lexical, title, char,
                lexical * (0.1 + geo),
                lexical * (0.1 + micro) * (0.2 + geo),
                lexical + 0.25 * filter_score]
    idx = np.unique(np.concatenate([top_indices(np.where(valid, s, -100), pool_size)
                                    for s in channels]))
    # если категория мала, добираем объявления из общего поиска, не возвращаем пустой список
    idx = idx[valid[idx]]
    if len(idx) < 50:
        idx = np.unique(np.concatenate([idx, top_indices(lexical, 100)]))
    rating_filter = bool(re.search(r'рейтинг.*4', str(row['search_infm_params_text']).lower()))
    rating_ok = ((item_ratings[idx] >= 4) | (item_ratings[idx] == 0)).astype(np.float32) if rating_filter else np.ones(len(idx), dtype=np.float32)
    features = np.column_stack([title[idx], description[idx], char[idx], params[idx],
                                filter_score[idx], geo[idx], micro[idx],
                                item_reviews[idx], rating_ok]).astype(np.float32)
    return idx, features

def score_candidates(features, config):
    title, description, char, params, filters, geo, micro, reviews, rating_ok = features.T
    lexical = (config['title'] * title + config['description'] * description
               + config['char'] * char + config.get('params', 0) * params)
    score = lexical * (1 + config['geo'] * geo) * (1 + config['micro'] * micro)
    score += config['filters'] * filters * (0.2 + geo)
    score += config.get('reviews', 0) * reviews * np.minimum(lexical, 0.5)
    score *= 0.85 + 0.15 * rating_ok
    return score

def build_candidate_cache(frame, desc):
    result = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=desc):
        idx, features = get_candidates(row)
        relevant = {item_to_idx[x] for x in row['item_id']}
        result.append((idx, features, relevant))
    return result

def evaluate_cached(cache, config, k=50):
    recalls = []
    for idx, features, relevant in cache:
        pred = set(idx[top_indices(score_candidates(features, config), k)])
        recalls.append(len(pred & relevant) / len(relevant))
    return float(np.mean(recalls)), recalls

def ranker_features(features, config):
    base = score_candidates(features, config)
    result = [features, base[:, None]]
    # относительное сходство и максимумы помогают сравнивать сложные и простые запросы
    for col in [0, 1, 2, 4, 6]:
        values = features[:, col]
        result.append((values / max(values.max(), 1e-6))[:, None])
    result.append((base / max(base.max(), 1e-6))[:, None])
    result.append(np.tile(np.max(features[:, :3], axis=0), (len(features), 1)))
    return np.column_stack(result).astype(np.float32)

def prepare_ranker_train(cache, config):
    x_parts, y_parts, weight_parts = [], [], []
    rng = np.random.default_rng(RANDOM_STATE)
    for idx, features, relevant in cache:
        labels = np.isin(idx, list(relevant)).astype(np.int32)
        if not labels.any():
            continue
        hard = top_indices(score_candidates(features, config), 140)
        random_idx = rng.choice(len(idx), size=min(50, len(idx)), replace=False)
        selected = np.unique(np.concatenate([hard, random_idx, np.where(labels)[0]]))
        x_parts.append(ranker_features(features, config)[selected])
        y_parts.append(labels[selected])
        # каждый положительный пример весит 1 / число positives запроса — как в macro Recall
        weight_parts.append(np.full(len(selected), 1 / len(relevant), dtype=np.float32))
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(weight_parts)

def evaluate_ranker(cache, model, config):
    recalls = []
    for idx, features, relevant in cache:
        scores = model.predict_proba(ranker_features(features, config))[:, 1]
        pred = set(idx[top_indices(scores, 50)])
        recalls.append(len(pred & relevant) / len(relevant))
    return float(np.mean(recalls))

# %% markdown
#### **Подбираем веса на небольшой выборке и проверяем на holdout**

Это Recall@50 только по известным положительным объявлениям, доступным в benchmark_items. Он не равен скрытой метрике организаторов и не сопоставим напрямую со старым Recall по недоступным объявлениям. Holdout не используется при подборе весов.

# %% code
tune_cache = build_candidate_cache(tune_sample, 'Кандидаты для подбора')
configs = []
for title_weight, desc_weight, char_weight in [(0.5, 0.35, 0.15), (0.35, 0.5, 0.15),
                                               (0.6, 0.2, 0.2), (0.35, 0.3, 0.35)]:
    for geo_weight in [1, 3, 7, 15]:
        for micro_weight in [0, 0.5, 1.5]:
            configs.append(dict(title=title_weight, description=desc_weight, char=char_weight,
                                geo=geo_weight, micro=micro_weight, filters=0.08, reviews=0.02))
results = [(evaluate_cached(tune_cache, config)[0], config) for config in configs]
results.sort(key=lambda x: x[0], reverse=True)
best_recall, best_config = results[0]
print('Лучшие варианты на подборе:')
for score, config in results[:5]:
    print(round(score, 4), config)

rank_train_cache = build_candidate_cache(rank_train_sample, 'Кандидаты для обучения ранкера')
x_rank, y_rank, w_rank = prepare_ranker_train(rank_train_cache, best_config)
ranker = CatBoostClassifier(iterations=650, depth=6, learning_rate=0.045,
                           loss_function='Logloss', l2_leaf_reg=6,
                           thread_count=6, random_seed=RANDOM_STATE, verbose=200,
                           allow_writing_files=False)
print('Обучаем небольшой ранкер:', x_rank.shape, 'positives:', y_rank.sum(), flush=True)
ranker.fit(x_rank, y_rank, sample_weight=w_rank)
ranker_tune_recall = evaluate_ranker(tune_cache, ranker, best_config)
use_ranker = ranker_tune_recall > best_recall + 0.005
print(f'Подбор: формула = {best_recall:.4f}, CatBoost = {ranker_tune_recall:.4f}; используем CatBoost: {use_ranker}')
# выбор делаем только на tune; holdout остается для независимой оценки
if use_ranker:
    x_rank, y_rank, w_rank = prepare_ranker_train(rank_train_cache + tune_cache, best_config)
    ranker.fit(x_rank, y_rank, sample_weight=w_rank)
    ranker.save_model(str(OUTPUT_DIR / 'ranker.cbm'))
del x_rank, y_rank, w_rank
gc.collect()

eval_cache = build_candidate_cache(eval_sample, 'Кандидаты holdout')
recall_formula, recalls_formula = evaluate_cached(eval_cache, best_config)
recall_final = evaluate_ranker(eval_cache, ranker, best_config) if use_ranker else recall_formula
pool_recall = np.mean([len(set(idx) & relevant) / len(relevant) for idx, _, relevant in eval_cache])
print(f'>>> Recall@50 формулы на holdout: {recall_formula:.4f}')
print(f'>>> Recall@50 на holdout: {recall_final:.4f}')
print(f'>>> Recall пула кандидатов на holdout: {pool_recall:.4f}')
metrics = {'tune_recall_at_50': best_recall, 'holdout_recall_at_50': recall_final,
           'tune_ranker_recall_at_50': ranker_tune_recall, 'use_ranker': bool(use_ranker),
           'holdout_formula_recall_at_50': recall_formula, 'n_rank_train': len(rank_train_sample),
           'holdout_pool_recall': float(pool_recall), 'config': best_config,
           'n_tune': len(tune_sample), 'n_holdout': len(eval_sample),
           'old_metric_ceiling': float(coverage_ceiling),
           'note': 'Recall по доступным positives; не скрытый benchmark Recall.'}
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
joblib.dump({'tune': tune_cache, 'holdout': eval_cache, 'rank_train': rank_train_cache,
             'config': best_config}, 'work/validation_cache.joblib', compress=0)

# %% markdown
#### **Поиск топ-50 для benchmark и сохранение ответа**

После оценки переобучаем только быстрый prior на всем train. Основные текстовые индексы остаются теми же. В выходном файле ровно 50 уникальных объявлений на каждый query_id; все item_id принадлежат benchmark_items. CSV и Parquet содержат одинаковую таблицу пар.

# %% code
prior_vectorizer, prior_model, location_probs = fit_priors(train)
microcat_columns = {int(value): idx for idx, value in enumerate(prior_model.classes_)}
geo_scores.cache_clear()

predictions = []
for _, row in tqdm(queries.iterrows(), total=len(queries), desc='Benchmark top-50'):
    idx, features = get_candidates(row)
    scores = ranker.predict_proba(ranker_features(features, best_config))[:, 1] if use_ranker else score_candidates(features, best_config)
    selected = idx[top_indices(scores, 50)]
    predictions.extend((row['query_id'], item_ids[j]) for j in selected)

submission = pd.DataFrame(predictions, columns=['query_id', 'item_id'])
assert len(submission) == len(queries) * 50
assert submission.groupby('query_id')['item_id'].nunique().eq(50).all()
assert not submission.duplicated(['query_id', 'item_id']).any()
assert set(submission['query_id']) == set(queries['query_id'])
assert submission['item_id'].isin(items['item_id']).all()
assert not submission.isna().any().any()
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
submission.to_parquet(OUTPUT_DIR / 'submission.parquet', index=False)
print(submission.head(10).to_string(index=False))
print('Сохранено строк:', len(submission), 'запросов:', submission['query_id'].nunique())
print('Файлы:', OUTPUT_DIR / 'submission.csv', OUTPUT_DIR / 'submission.parquet')
print('Полное время, минут:', round((time.time() - START_TIME) / 60, 2))

# %% markdown
#### **Библиотеки и воспроизводимость**

Использованы open-source библиотеки pandas, NumPy, SciPy, scikit-learn, NLTK (Russian SnowballStemmer), CatBoost, tqdm, joblib и pyarrow. Внешние API, нейросетевые веса и ручные ответы под query_id не используются. Для воспроизведения: установить зависимости, положить три исходных Parquet в dataset и выполнить ячейки сверху вниз. Случайное разбиение фиксировано random_state=42. Невыбранные объявления в обучении ранкера используются как слабые отрицательные примеры: часть из них может быть релевантной, но не отмеченной в train.
