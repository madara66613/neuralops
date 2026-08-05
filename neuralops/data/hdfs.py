"""Official Loghub HDFS v1 download and block-sequence ingestion."""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import shutil
import ssl
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import certifi

from neuralops.data.records import SequenceRecord, validate_record

HDFS_V1_URL = "https://zenodo.org/api/records/8196385/files/HDFS_v1.zip/content"
HDFS_V1_MD5 = "76a24b4d9a6164d543fb275f89773260"
BLOCK_PATTERN = re.compile(r"blk_-?\d+")
ATTRIBUTION = """Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics
Curated by LOGPAI. DOI: 10.5281/zenodo.8196385
Dataset license: CC BY 4.0
Repository: https://github.com/logpai/loghub

Please cite the Loghub ISSRE 2023 paper where applicable. This notice must be
included in copies of the Loghub datasets. NeuralOps source code is separately
licensed under MIT and does not relicense this dataset.
"""


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(target)


def download_hdfs_v1(target_dir: Path, *, force: bool = False) -> Path:
    """Download, verify, and safely extract the official HDFS v1 archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / "HDFS_v1.zip"
    if archive.exists() and not force and md5_file(archive) == HDFS_V1_MD5:
        pass
    else:
        temporary = archive.with_suffix(".zip.part")
        request = urllib.request.Request(HDFS_V1_URL, headers={"User-Agent": "NeuralOps/0.1"})
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with (
            urllib.request.urlopen(request, timeout=120, context=tls_context) as response,
            temporary.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if md5_file(temporary) != HDFS_V1_MD5:
            temporary.unlink(missing_ok=True)
            raise ValueError("HDFS v1 checksum mismatch; partial file removed")
        temporary.replace(archive)

    if md5_file(archive) != HDFS_V1_MD5:
        raise ValueError(f"Existing archive has unexpected checksum: {archive}")
    extracted = target_dir / "extracted"
    if force and extracted.exists():
        raise ValueError("Refusing to overwrite extracted data; remove it explicitly first")
    if not extracted.exists():
        _safe_extract(archive, extracted)
    (target_dir / "UPSTREAM_ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    return extracted


def _find_one(root: Path, candidates: tuple[str, ...]) -> Path | None:
    matches = [path for name in candidates for path in root.rglob(name)]
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(f"Multiple candidate files found for {candidates}: {unique}")
    return unique[0] if unique else None


def _read_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"BlockId", "Label"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected BlockId and Label columns in {path}")
        for row in reader:
            block_id = row["BlockId"].strip()
            raw_label = row["Label"].strip().lower()
            if raw_label not in {"normal", "anomaly"}:
                raise ValueError(f"Unknown HDFS label {row['Label']!r}")
            labels[block_id] = int(raw_label == "anomaly")
    return labels


def _parse_event_sequence(value: str) -> tuple[str, ...]:
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # The full official HDFS archive uses unquoted IDs: [E5,E22,E5].
            events = tuple(item.strip() for item in text[1:-1].split(",") if item.strip())
        else:
            if not isinstance(parsed, (list, tuple)):
                raise ValueError("EventSequence literal must contain a list")
            events = tuple(str(item).strip() for item in parsed if str(item).strip())
    else:
        events = tuple(item for item in re.split(r"[\s,]+", text) if item)
    if not events:
        raise ValueError("Empty HDFS event sequence")
    return events


def _read_traces(path: Path) -> dict[str, tuple[str, ...]]:
    traces: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "BlockId" not in reader.fieldnames:
            raise ValueError(f"Expected BlockId column in {path}")
        sequence_column = "EventSequence" if "EventSequence" in reader.fieldnames else "Features"
        if sequence_column not in reader.fieldnames:
            raise ValueError(f"Expected EventSequence or Features column in {path}")
        for row in reader:
            block_id = row["BlockId"].strip()
            traces[block_id] = _parse_event_sequence(row[sequence_column])
    return traces


def _read_structured(path: Path) -> dict[str, tuple[str, ...]]:
    csv.field_size_limit(64 * 1024 * 1024)
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"Content", "EventId"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected Content and EventId columns in {path}")
        for row in reader:
            event_id = row["EventId"].strip()
            if not event_id:
                continue
            for block_id in dict.fromkeys(BLOCK_PATTERN.findall(row["Content"])):
                grouped[block_id].append(event_id)
    return {block_id: tuple(events) for block_id, events in grouped.items()}


def load_hdfs_records(root: Path) -> list[SequenceRecord]:
    """Load official preprocessed traces or aggregate an official structured CSV."""
    labels_path = _find_one(root, ("anomaly_label.csv",))
    if labels_path is None:
        raise FileNotFoundError(f"Could not find anomaly_label.csv below {root}")
    traces_path = _find_one(root, ("Event_traces.csv",))
    structured_path = _find_one(root, ("HDFS.log_structured.csv", "HDFS_2k.log_structured.csv"))
    if traces_path is not None:
        traces = _read_traces(traces_path)
    elif structured_path is not None:
        traces = _read_structured(structured_path)
    else:
        raise FileNotFoundError("Could not find Event_traces.csv or HDFS structured CSV")

    labels = _read_labels(labels_path)
    missing_labels = sorted(set(traces) - set(labels))
    if missing_labels:
        raise ValueError(f"Missing labels for {len(missing_labels)} HDFS blocks")
    missing_traces = sorted(set(labels) - set(traces))
    if missing_traces:
        raise ValueError(f"Missing traces for {len(missing_traces)} labeled HDFS blocks")

    records = [
        SequenceRecord(
            session_id=block_id,
            group_id=block_id,
            events=traces[block_id],
            anomaly=labels[block_id],
            source="loghub-hdfs-v1",
        )
        for block_id in sorted(traces)
    ]
    for record in records:
        validate_record(record)
    return records
