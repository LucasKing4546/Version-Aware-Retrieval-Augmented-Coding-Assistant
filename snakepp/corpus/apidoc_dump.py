"""Dump public API docstrings of installed modules as JSON lines. Standard library only.

It runs inside a version-specific environment (data/envs/docs/<library>-<version>), not in the project venv:
    python apidoc_dump.py OUT.jsonl ROOT MODULE [MODULE ...]
ROOT is the package name objects must come from (e.g. "pandas"), so re-exported numpy objects are skipped.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import warnings


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _own(obj, root: str) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return module == root or module.startswith(root + ".")


def _public(names) -> list[str]:
    return [n for n in names if not n.startswith("_")]


def iter_api(module_name: str, root: str):
    module = importlib.import_module(module_name)
    names = getattr(module, "__all__", None) or dir(module)
    for name in _public(names):
        obj = getattr(module, name, None)
        if obj is None or inspect.ismodule(obj) or not (callable(obj) and _own(obj, root)):
            continue
        yield f"{module_name}.{name}", obj
        if inspect.isclass(obj):
            for member in _public(dir(obj)):
                attr = inspect.getattr_static(obj, member, None)
                target = getattr(obj, member, None)
                if isinstance(attr, property) or (callable(target) and _own(target, root)):
                    yield f"{module_name}.{name}.{member}", attr if isinstance(attr, property) else target


def main(out_path: str, root: str, modules: list[str]) -> None:
    warnings.simplefilter("ignore")
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for module_name in modules:
            for qualname, obj in iter_api(module_name, root):
                doc = inspect.getdoc(obj)
                if qualname in seen or not doc:
                    continue
                seen.add(qualname)
                sig = "" if isinstance(obj, property) else _signature(obj)
                out.write(json.dumps({"qualname": qualname, "signature": sig, "doc": doc}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
