# run_config.py
from typing import Any

def cfg_get(obj: Any, key: str | tuple | list, default=None):
    """
    Robust getter for config-like objects.
    Checks:
      1) attribute on obj
      2) key in dict(obj) if mapping
      3) attribute on obj.crl (if present)
      4) 'dotted' paths like 'data.panel_path'
      5) alias list/tuple: first present wins
    """
    def _one(o, k):
        # dotted path
        if isinstance(k, str) and "." in k:
            cur = o
            for part in k.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                elif isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur
        # flat attr / key
        if hasattr(o, k):
            return getattr(o, k)
        if isinstance(o, dict) and k in o:
            return o[k]
        # check nested .crl
        if hasattr(o, "crl"):
            c = getattr(o, "crl")
            if hasattr(c, k):
                return getattr(c, k)
            if isinstance(c, dict) and k in c:
                return c[k]
        return None

    if isinstance(key, (tuple, list)):
        for k in key:
            v = _one(obj, k)
            if v is not None:
                return v
        return default
    v = _one(obj, key)
    return default if v is None else v

def cfg_get_nested(obj: Any, dotted: str, default=None):
    v = cfg_get(obj, dotted, None)
    return default if v is None else v