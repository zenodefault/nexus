from typing import Dict, Iterable, List


class SchemaError(ValueError):
    pass


def require_fields(data: Dict, fields: Iterable[str], name: str) -> None:
    missing: List[str] = [f for f in fields if f not in data]
    if missing:
        raise SchemaError(f"{name} missing fields: {', '.join(missing)}")
