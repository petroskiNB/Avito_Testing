import json
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

data = joblib.load('work/validation_cache.joblib')
config = data['config']
plain_recall = np.mean([len(set(idx[np.argsort(-(0.35*f[:, 0] + 0.3*f[:, 1] + 0.35*f[:, 2]))[:50]]) & rel) / len(rel)
                       for idx, f, rel in data['holdout']])
print('Text-only holdout Recall@50:', plain_recall, flush=True)

def base_score(f):
    t, d, c, p, filters, geo, micro, reviews, rating = f.T
    lexical = config['title'] * t + config['description'] * d + config['char'] * c
    score = lexical * (1 + config['geo'] * geo) * (1 + config['micro'] * micro)
    score += config['filters'] * filters * (0.2 + geo)
    score += config['reviews'] * reviews * np.minimum(lexical, 0.5)
    return score * (0.85 + 0.15 * rating)

def features(f):
    base = base_score(f)
    out = [f, base[:, None]]
    # относительная позиция объявления важна вместе с абсолютным сходством
    for col in [0, 1, 2, 4, 6]:
        x = f[:, col]
        out.append((x / max(x.max(), 1e-6))[:, None])
    out.append((base / max(base.max(), 1e-6))[:, None])
    out.append(np.tile(np.max(f[:, :3], axis=0), (len(f), 1)))
    return np.column_stack(out).astype(np.float32)

def make_train(cache):
    xs, ys, ws = [], [], []
    rng = np.random.default_rng(42)
    for idx, f, relevant in cache:
        y = np.isin(idx, list(relevant)).astype(np.int32)
        if not y.any():
            continue
        hard = np.argsort(-base_score(f))[:140]
        other = rng.choice(len(idx), size=min(50, len(idx)), replace=False)
        selected = np.unique(np.concatenate([hard, other, np.where(y)[0]]))
        xs.append(features(f)[selected])
        ys.append(y[selected])
        ws.append(np.full(len(selected), 1 / len(relevant), dtype=np.float32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)

def evaluate(model, cache):
    recalls = []
    for idx, f, relevant in cache:
        scores = model.predict_proba(features(f))[:, 1]
        pred = set(idx[np.argsort(-scores)[:50]])
        recalls.append(len(pred & relevant) / len(relevant))
    return float(np.mean(recalls))

train_cache, tune_cache = data['tune'][:500], data['tune'][500:]
x, y, w = make_train(train_cache)
print('Ranker train:', x.shape, 'positives:', y.sum(), flush=True)
t0 = time.time()
model = CatBoostClassifier(iterations=450, depth=6, learning_rate=0.045,
                           loss_function='Logloss', l2_leaf_reg=6,
                           thread_count=6, random_seed=42, verbose=100,
                           allow_writing_files=False)
model.fit(x, y, sample_weight=w)
ranker_score = evaluate(model, tune_cache)
base = np.mean([len(set(idx[np.argsort(-base_score(f))[:50]]) & rel) / len(rel) for idx, f, rel in tune_cache])
print('Selection subset baseline / ranker:', base, ranker_score, flush=True)
if ranker_score > base + 0.005:
    x, y, w = make_train(data['tune'])
    model.fit(x, y, sample_weight=w)
    score = evaluate(model, data['holdout'])
    print('Ranker final holdout:', score, 'elapsed:', time.time() - t0, flush=True)
    model.save_model('work/ranker.cbm')
    with open('work/ranker_metrics.json', 'w') as f:
        json.dump({'selection_baseline': base, 'selection_ranker': ranker_score, 'holdout_ranker': score}, f, indent=2)
else:
    print('Ranker rejected; keep baseline', flush=True)
