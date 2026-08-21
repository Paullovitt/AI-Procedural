from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {'.git','__pycache__','gpu_cache','model','rigorous_results','rigorous_results_v2','rigorous_results_v12'}
FORBIDDEN_IMPORT_PREFIXES = ('tensorflow','keras')
FORBIDDEN_ATTR_CHAINS = {
    'torch.nn','torch.autograd','torch.optim','torch.Tensor.backward','torch.backward'
}


def attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def scan_file(path: Path):
    problems = []
    try:
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f'{path.name}: parse error: {exc}']
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    problems.append(f'{path.name}:{node.lineno}: forbidden import {alias.name}')
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.startswith(FORBIDDEN_IMPORT_PREFIXES) or mod.startswith(('torch.nn','torch.autograd','torch.optim')):
                problems.append(f'{path.name}:{node.lineno}: forbidden import {mod}')
        elif isinstance(node, ast.Attribute):
            chain = attr_chain(node)
            if chain in FORBIDDEN_ATTR_CHAINS or chain.startswith(('torch.nn.','torch.autograd.','torch.optim.')):
                problems.append(f'{path.name}:{node.lineno}: forbidden API {chain}')
        elif isinstance(node, ast.Call):
            chain = attr_chain(node.func) if isinstance(node.func, ast.Attribute) else ''
            if chain.endswith('.backward'):
                problems.append(f'{path.name}:{node.lineno}: backward() is forbidden')
            for kw in node.keywords:
                if kw.arg == 'requires_grad' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    problems.append(f'{path.name}:{node.lineno}: requires_grad=True is forbidden')
    return problems


def main():
    problems = []
    checked = 0
    for path in ROOT.rglob('*.py'):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1]):
            continue
        checked += 1
        problems.extend(scan_file(path))
    if problems:
        print('ARCHITECTURE GUARD: FAIL')
        for row in problems:
            print(' -', row)
        raise SystemExit(1)
    print(f'ARCHITECTURE GUARD: OK ({checked} Python files checked)')


if __name__ == '__main__':
    main()
