import json
import shutil
import zipfile
from pathlib import Path

out = Path('outputs')
metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
requirements = '''pandas>=2.2,<4
numpy>=1.26,<3
scipy>=1.12,<2
scikit-learn>=1.5,<2
nltk>=3.9,<4
pyarrow>=16
tqdm>=4.66
joblib>=1.4
catboost==1.2.8
'''
(out / 'requirements.txt').write_text(requirements, encoding='utf-8')
description = f'''# Решение задачи кандидатогенерации

## Файлы

- `1.ipynb` — полный код с комментариями и результатами выполнения.
- `submission.parquet` и `submission.csv` — одна и та же таблица: `query_id`, `item_id`.
  Для каждого из 2452 запросов выдано ровно 50 уникальных объявлений: всего 122600 строк.
- `metrics.json` — локальные метрики и параметры.
- `requirements.txt` — зависимости.

На приложенных скриншотах указан смысл ответа (до 50 item_id на запрос), но не задана точная схема файла загрузки. Поэтому сохранены два распространенных формата одной таблицы. При наличии отдельного шаблона платформы требуется только преобразование формата, не новый поиск.

## Подход

1. Очистка HTML, нормализация ё/е и русский SnowballStemmer.
2. Отдельные BM25-индексы для заголовков, описаний и параметров.
3. Символьный TF-IDF по заголовкам (3–5 символов), устойчивый к части опечаток.
4. Мягкий географический приоритет: совпадение локации, частоты переходов между локациями в train и приближенное расстояние до центра города.
5. Быстрый MultinomialNB для вероятности подкатегории по запросу. Запросы валидации исключены из обучения.
6. Объединение нескольких списков кандидатов и отбор 50. Категория 0 не ограничивает поиск.

Открытые библиотеки перечислены в requirements.txt. Внешних API, загрузки моделей при инференсе и ручных ответов под query_id нет.

## Проверка

В исходном корпусе есть объявления только для 6.63% строк train. Поэтому Recall по всем train-позитивам имеет искусственный потолок около {metrics['old_metric_ceiling']:.4f}.
Для локальной оценки оставлены известные положительные объявления, находящиеся в корпусе. Это меняет постановку метрики: результат не следует напрямую сравнивать со старым Recall и со скрытым benchmark.

Подбор: {metrics['n_tune']} запросов. Отложенная проверка: {metrics['n_holdout']} запросов. Нормализованные тексты запросов не пересекаются между подбором, holdout и обучением prior.

- Recall@50 на отложенных запросах: **{metrics['holdout_recall_at_50']:.4f}**.
- Recall объединенного пула кандидатов: **{metrics['holdout_pool_recall']:.4f}**.

Проверены число строк, полнота query_id, уникальность пар, 50 разных item_id на запрос, отсутствие пропусков и принадлежность всех item_id корпусу.

## Воспроизведение

Рекомендуется Python 3.12. Установить зависимости: `python -m pip install -r requirements.txt`.
Положить `train.parquet`, `benchmark_queries.parquet`, `benchmark_items.parquet` в папку `dataset` рядом с ноутбуком и выполнить ячейки сверху вниз. Папки `work` и `outputs` создаются автоматически. Исходные данные в архив решения не включены.
'''
if metrics.get('use_ranker'):
    description += '\nДля финального отбора дополнительно используется небольшой CatBoost по признакам сходства и географии; обучение и выбор варианта полностью включены в ноутбук.\n'
elif 'tune_ranker_recall_at_50' in metrics:
    description += f"\nДополнительно проверен CatBoost на {metrics['n_rank_train']} обучающих запросах. На подборе он дал {metrics['tune_ranker_recall_at_50']:.4f} против {metrics['tune_recall_at_50']:.4f} у формулы. Прирост меньше заранее заданного порога 0.005, поэтому итоговый ответ использует формулу. Код эксперимента сохранен в ноутбуке.\n"
(out / 'README.md').write_text(description, encoding='utf-8')
shutil.copy2('1.ipynb', out / '1.ipynb')
with zipfile.ZipFile(out / 'solution.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['1.ipynb', 'submission.parquet', 'submission.csv', 'metrics.json', 'requirements.txt', 'README.md']:
        archive.write(out / name, arcname=name)
    if metrics.get('use_ranker') and (out / 'ranker.cbm').exists():
        archive.write(out / 'ranker.cbm', arcname='ranker.cbm')
print('Saved', out / 'solution.zip')
