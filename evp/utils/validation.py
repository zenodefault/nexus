from typing import Dict, Iterable, List


class SchemaError(ValueError):
    pass


def fill_missing(data: Dict, defaults: Dict) -> Dict:
    out = dict(data or {})
    for key, value in defaults.items():
        if key not in out:
            out[key] = value
    return out


def require_fields(data: Dict, fields: Iterable[str], name: str) -> None:
    missing: List[str] = [f for f in fields if f not in data]
    if missing:
        raise SchemaError(f"{name} missing fields: {', '.join(missing)}")
