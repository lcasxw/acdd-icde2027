import os, csv, json, hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
PROVENANCE=BASE/'units.json'
TARGET_ROOT=BASE/'inputs'

def local_path(path):
    absolute = str(Path(path).resolve())
    return Path("\\\\?\\" + absolute) if os.name == "nt" and not absolute.startswith("\\\\?\\") else Path(absolute)

def sha(path):
    digest = hashlib.sha256()
    with local_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return local_path(path).read_text(encoding="utf-8")

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
