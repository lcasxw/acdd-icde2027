"""Build the published score-replay inputs from the repository and the original BCI labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "engine"))
from public_paths import system_path

BASE = system_path(BASE)
LABEL_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"
LABEL_SHA = "a5e21dbd7e92d2959a3579d18de0defa9ca1a85eec17931118f81dbe141c6302"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def original_labels(local_copy):
    if local_copy is None:
        request = Request(LABEL_URL, headers={"User-Agent": "ACDD-reproduction/1"})
        with urlopen(request, timeout=60) as response:
            data = response.read(100_000)
    else:
        data = local_copy.read_bytes()
    if hashlib.sha256(data).hexdigest() != LABEL_SHA:
        raise ValueError("The original BCI Competition IV 2a label archive has a different digest")
    result = {}
    with ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            mat = archive.read(name)
            if mat[126:128] != b"IM" or struct.unpack_from("<I", mat, 128)[0] != 15:
                raise ValueError("Unexpected MATLAB file in the pinned label archive")
            content = zlib.decompress(mat[136:])
            if struct.unpack_from("<II", content) != (14, len(content) - 8):
                raise ValueError("Unexpected MATLAB matrix")
            fields = {}
            pos = 8
            while pos < len(content):
                kind, size = struct.unpack_from("<II", content, pos)
                fields[kind] = content[pos + 8:pos + 8 + size]
                pos += 8 + ((size + 7) // 8) * 8
            if fields[1] != b"classlabel" or len(fields[2]) != 288:
                raise ValueError("Expected the original 288 trial labels")
            labels = [label - 1 for label in fields[2] if label in (1, 2)]
            if len(labels) != 144:
                raise ValueError("Expected 144 left/right trials in each session")
            result[name] = labels
    return result


def restore_bnci_rows(content, labels):
    # Keep delimiters and line endings exactly as exported. Only the empty label field is filled.
    lines = content.splitlines(keepends=True)
    header = lines[0].decode("ascii").strip().split(",")
    col = {name: header.index(name) for name in ("dataset_id", "subject_key", "session_id", "run_id", "y_true")}
    count = {}
    for i in range(1, len(lines)):
        row = lines[i].split(b",")
        if row[col["dataset_id"]] != b"BNCI2014_001":
            continue
        if row[col["y_true"]] != b"":
            raise ValueError("The public payload must leave the BNCI cue cell empty")
        subject = int(row[col["subject_key"]].split(b"::")[1])
        session = {b"0train": "T", b"1test": "E"}[row[col["session_id"]]]
        key = f"A{subject:02d}{session}.mat"
        index = count.get(key, 0)
        if int(row[col["run_id"]]) != index // 24:
            raise ValueError("BNCI run order differs from the recorded export")
        row[col["y_true"]] = str(labels[key][index]).encode("ascii")
        count[key] = index + 1
        lines[i] = b",".join(row)
    if not count or any(n != 144 for n in count.values()):
        raise ValueError("Incomplete BNCI validation session")
    return b"".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=BASE.parent / "input-volumes")
    parser.add_argument("--labels", type=Path, help="Optional original true_labels.zip for offline use")
    parser.add_argument("--out", type=Path, default=BASE / "inputs")
    args = parser.parse_args()
    index = json.loads((BASE / "public_input_layout.json").read_bytes())
    with (BASE / "required_inputs.csv").open(encoding="utf-8", newline="") as f:
        expected = {r["path"]: r["sha256"] for r in csv.DictReader(f)}
    destination = system_path(args.out.resolve())
    args.archives = system_path(args.archives)
    if destination.exists():
        parser.error("Choose a new output directory, or verify the existing inputs with reproduce.py verify-inputs")
    labels = original_labels(args.labels)
    restore = set(index["sources_with_upstream_labels"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".acdd-inputs-", dir=destination.parent) as temporary:
        tree = Path(temporary) / "inputs"
        seen = set()
        for volume in index["volumes"]:
            path = args.archives / volume["filename"]
            if file_hash(path) != volume["sha256"]:
                raise ValueError(f"Volume digest mismatch: {path.name}")
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    if name not in expected or name in seen or name != PurePosixPath(name).as_posix() or set(PurePosixPath(name).parts) & {"..", "."} or PurePosixPath(name).is_absolute() or "\\" in name or ":" in name:
                        raise ValueError(f"Unexpected or duplicate input: {name}")
                    seen.add(name)
                    out = tree / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, out.open("xb") as sink:
                        if name in restore:
                            sink.write(restore_bnci_rows(source.read(), labels))
                        else:
                            shutil.copyfileobj(source, sink)
                    if file_hash(out) != expected[name]:
                        raise ValueError(f"Materialized input digest mismatch: {name}")
        if seen != set(expected) or len(seen) != 363 or not restore.issubset(seen):
            raise ValueError("The materialized set must cover all 363 entries")
        tree.rename(destination)
    print(f"PASS: materialized and verified all {len(seen)} public replay inputs")


if __name__ == "__main__":
    main()
