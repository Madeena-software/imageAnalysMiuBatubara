import json
import re
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[1]

notebooks = {
    'block': ROOT / 'block-detection.ipynb',
    'circle': ROOT / 'circle-detection.ipynb',
}

prods = {
    'block': ROOT / 'public' / 'image-analysis-miu-batubara' / 'block_detection.py',
    'circle': ROOT / 'public' / 'image-analysis-miu-batubara' / 'circle_detection.py',
}

functions_to_check = {
    'block': ['_load_and_validate_image', '_shrink_box', '_mean_intensity_in_box'],
    'circle': ['_load_and_validate_image'],
}


def extract_from_py(path, func_name):
    text = path.read_text()
    # naive regex to capture def func(...): ... (until next def or EOF)
    pattern = r"(^def\s+%s\b[\s\S]*?)(?=^def\s+|\Z)" % re.escape(func_name)
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def extract_from_notebook(nb_path, func_name):
    nb = json.loads(nb_path.read_text())
    code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    src = '\n'.join('\n'.join(c.get('source', [])) for c in code_cells)
    pattern = r"(^def\s+%s\b[\s\S]*?)(?=^def\s+|\Z)" % re.escape(func_name)
    m = re.search(pattern, src, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def normalize_code(s):
    if s is None:
        return None
    # Remove trailing spaces, normalize indentation, strip leading/trailing blank lines
    lines = s.splitlines()
    # remove leading/trailing blank
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    # strip trailing spaces
    lines = [re.sub(r"\s+$", "", l) for l in lines]
    return '\n'.join(lines)


def main():
    ok = True
    for key in ['block', 'circle']:
        print('\n=== Comparing notebook -> production for', key, '===')
        nb = notebooks[key]
        py = prods[key]
        for fn in functions_to_check[key]:
            nb_code = extract_from_notebook(nb, fn)
            py_code = extract_from_py(py, fn)
            n_nb = normalize_code(nb_code)
            n_py = normalize_code(py_code)
            print('\nFunction:', fn)
            if n_nb is None:
                print('  - NOT FOUND in notebook')
                ok = False
                continue
            if n_py is None:
                print('  - NOT FOUND in production module')
                ok = False
                continue
            if n_nb == n_py:
                print('  - MATCH: code is identical')
            else:
                print('  - DIFFER: showing unified diff (notebook vs production)')
                diff = difflib.unified_diff(
                    n_py.splitlines(), n_nb.splitlines(),
                    fromfile=str(py), tofile=str(nb), lineterm=''
                )
                for line in diff:
                    print(line)
                ok = False
    if ok:
        print('\nAll checked functions match exactly.')
        return 0
    else:
        print('\nSome functions differ. Please review diffs above.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
