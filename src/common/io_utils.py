"""Deterministic JSON/JSONL I/O and cryptographic hashing utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(p)
    return count


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_key(obj: Any) -> str:
    """Deterministic SHA-256 key using Python's default JSON separators (matches annotation filenames)."""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_rank(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def load_heldout_parents(policy_dir: str | Path, extra_paths: Iterable[str | Path] = ()) -> set[str]:
    """Collect all held-out parent IDs from dev/valid/test splits and optional evaluation files."""
    pdir = Path(policy_dir)
    held: set[str] = set()
    for name in ("esconv_dev.jsonl", "esconv_test.jsonl", "ed_valid.jsonl", "ed_test.jsonl"):
        for row in read_jsonl(pdir / name):
            if "parent_id" in row:
                held.add(str(row["parent_id"]))
    for extra in extra_paths:
        for row in read_jsonl(extra):
            if "parent_id" in row:
                held.add(str(row["parent_id"]))
    return held


# Compatibility aliases
sha = sha256_file
digest = sha256_file
key = canonical_json_key
