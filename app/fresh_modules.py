"""Reload shared app modules whose files changed on disk (D-105).

Streamlit re-runs the page scripts on every visit but keeps imported modules in
memory. After a code update on the hosted app, a page could then import new names
from an old copy of workflow_ui and fail with ImportError. Every page imports this
module first and calls refresh().
"""
import importlib
import os
import sys

# Dependency order: a module is reloaded after the modules it imports from.
NAMES = ('stethoai.app_core', 'stethoai.drive_store', 'lung_chart', 'workflow_ui')
_seen = {}


def _mtime(module):
    try:
        return os.path.getmtime(module.__file__)
    except (OSError, TypeError, AttributeError):
        return None


def refresh():
    """Reload any listed module whose file changed since it was last seen."""
    changed = False
    for name in NAMES:
        module = sys.modules.get(name)
        if module is None:
            continue
        stamp = _mtime(module)
        if changed or (name in _seen and stamp != _seen[name]):
            module = importlib.reload(module)
            changed = True
        _seen[name] = _mtime(module)


# Modules already in memory when this file is first imported predate it and may
# be stale copies from before an update: reload them once.
for _name in NAMES:
    _module = sys.modules.get(_name)
    if _module is not None:
        _seen[_name] = _mtime(importlib.reload(_module))
