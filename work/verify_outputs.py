import json
import zipfile
from pathlib import Path
import pandas as pd

out = Path('outputs')
csv = pd.read_csv(out / 'submission.csv', dtype=str)
parquet = pd.read_parquet(out / 'submission.parquet').astype(str)
assert csv.equals(parquet), 'CSV и Parquet различаются'
queries = pd.read_parquet('dataset/benchmark_queries.parquet', columns=['query_id'])
items = pd.read_parquet('dataset/benchmark_items.parquet', columns=['item_id'])
assert list(csv.columns) == ['query_id', 'item_id']
assert len(csv) == 50 * len(queries) == 122600
assert csv.groupby('query_id')['item_id'].nunique().eq(50).all()
assert not csv.duplicated().any()
assert set(csv.query_id) == set(queries.query_id)
assert set(csv.item_id) <= set(items.item_id)
nb = json.loads(Path('1.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell {i}', 'exec')
        assert cell['execution_count'] is not None, f'Не выполнена ячейка {i}'
        assert all(x['output_type'] != 'error' for x in cell['outputs']), f'Ошибка в ячейке {i}'
assert Path('1.ipynb').read_bytes() == (out / '1.ipynb').read_bytes()
with zipfile.ZipFile(out / 'solution.zip') as archive:
    assert archive.testzip() is None
    assert archive.read('1.ipynb') == (out / '1.ipynb').read_bytes()
print('PASS: CSV = Parquet; 2452 запроса x 50; все ID корректны; notebook выполнен; ZIP проверен.')
print((out / 'metrics.json').read_text(encoding='utf-8'))
