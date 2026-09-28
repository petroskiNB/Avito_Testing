import io
import json
import time
import traceback
from pathlib import Path
from contextlib import redirect_stdout

path = Path('1.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
count = 0
class Tee(io.StringIO):
    def write(self, s):
        import sys
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        return super().write(s)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'\nCELL {i} START {time.strftime("%H:%M:%S")}', flush=True)
    output = Tee()
    error = None
    try:
        with redirect_stdout(output):
            exec(compile(''.join(cell['source']), f'1.ipynb cell {i}', 'exec'), namespace)
    except BaseException as exc:
        error = exc
        traceback.print_exc()
    cell['execution_count'] = count
    cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue().splitlines(True)}]
    if error:
        cell['outputs'].append({'output_type': 'error', 'ename': type(error).__name__, 'evalue': str(error), 'traceback': traceback.format_exception(error)})
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    if error:
        raise error
    print(f'CELL {i} DONE {time.strftime("%H:%M:%S")}', flush=True)
