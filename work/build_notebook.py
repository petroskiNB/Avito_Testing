import json
from pathlib import Path

source = Path('work/solution_cells.py').read_text(encoding='utf-8')
cells = []
for block in source.split('# %% ')[1:]:
    kind, body = block.split('\n', 1)
    cell = {'cell_type': kind.strip(), 'metadata': {}, 'source': body.strip().splitlines(True)}
    if cell['cell_type'] == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
original = json.loads(Path('work/1.original.ipynb').read_text(encoding='utf-8'))
nb = {'cells': cells, 'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
Path('1.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook updated:', len(cells), 'cells')
