"""
media/extractor.py
Safe archive extraction with zip-slip protection, decompression-bomb
caps, optional nested recursion and per-file detection.

Supported: zip, tar (including .tar.gz / .tgz, tarfile handles gzip
internally), gzip. 7z and rar are recognised but raise ExtractionError
because no supported library is installed. The original archive is
never deleted or modified.
"""

from __future__ import annotations

import gzip
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from media.detector import DetectionResult, classify
from media.formats import (
    CATEGORY_ARCHIVE,
    PROCESSABLE_ARCHIVES,
    specs_for_filename,
    specs_for_header,
)


class ExtractionError(Exception):
    """Raised when an archive cannot be safely extracted."""


@dataclass(frozen=True)
class ExtractionLimits:
    max_files: int = 5000
    max_total_bytes: int = 2 * 1024 * 1024 * 1024       # 2 GiB
    max_single_file_bytes: int = 500 * 1024 * 1024      # 500 MiB
    max_depth: int = 2


DEFAULT_LIMITS = ExtractionLimits()
_CHUNK = 64 * 1024
_ARCHIVE_NAMES = PROCESSABLE_ARCHIVES | {"7z", "rar"}


def _archive_kind(path: Path) -> Optional[str]:
    lower = path.name.lower()
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    try:
        with open(path, "rb") as f:
            header = f.read(1024)
    except OSError:
        return None
    for spec in specs_for_header(header):
        if spec.name in _ARCHIVE_NAMES:
            return spec.name
    for spec in specs_for_filename(path.name):
        if spec.name in _ARCHIVE_NAMES:
            return spec.name
    return None


def _safe_target(root: Path, member_name: str) -> Path:
    root = root.resolve()
    clean = member_name.replace("\\", "/").lstrip("/")
    if len(clean) >= 2 and clean[1] == ":":
        clean = clean[2:].lstrip("/")
    if not clean or clean in (".", ".."):
        raise ExtractionError(f"unsafe path in archive: {member_name!r}")
    try:
        target = (root / clean).resolve()
    except OSError as exc:
        raise ExtractionError(f"cannot resolve {member_name!r}: {exc}") from exc
    try:
        target.relative_to(root)
    except ValueError:
        raise ExtractionError(
            f"zip-slip: {member_name!r} would escape {root}"
        ) from None
    return target


def _unique_dest(root: Path, base: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / base
    n = 1
    while candidate.exists():
        candidate = root / f"{base}_{n}"
        n += 1
    return candidate


def _unique_file(target: Path) -> Path:
    if not target.exists():
        return target
    n = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _extract_zip(
    archive_path: Path, target_dir: Path, limits: ExtractionLimits,
) -> List[Path]:
    extracted: List[Path] = []
    total_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > limits.max_files:
            raise ExtractionError(
                f"too many members in {archive_path.name}: "
                f"{len(infos)} > {limits.max_files}"
            )
        for info in infos:
            if info.is_dir():
                continue
            if info.file_size and info.file_size > limits.max_single_file_bytes:
                raise ExtractionError(
                    f"member too large: {info.filename} "
                    f"({info.file_size} > {limits.max_single_file_bytes})"
                )
            target = _unique_file(_safe_target(target_dir, info.filename))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > limits.max_total_bytes:
                            dst.close()
                            target.unlink(missing_ok=True)
                            raise ExtractionError(
                                "total extracted size exceeds "
                                f"{limits.max_total_bytes} bytes"
                            )
                        dst.write(chunk)
            except ExtractionError:
                raise
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                target.unlink(missing_ok=True)
                raise ExtractionError(
                    f"failed extracting {info.filename!r}: {exc}"
                ) from exc
            extracted.append(target)
    return extracted


def _extract_tar(
    archive_path: Path, target_dir: Path, limits: ExtractionLimits,
) -> List[Path]:
    extracted: List[Path] = []
    total_bytes = 0
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > limits.max_files:
            raise ExtractionError(
                f"too many members in {archive_path.name}: "
                f"{len(members)} > {limits.max_files}"
            )
        for member in members:
            if member.isdir():
                continue
            if member.issym() or member.islnk():
                raise ExtractionError(
                    f"refusing link in {archive_path.name}: {member.name}"
                )
            if not member.isfile():
                continue
            if member.size > limits.max_single_file_bytes:
                raise ExtractionError(
                    f"member too large: {member.name} "
                    f"({member.size} > {limits.max_single_file_bytes})"
                )
            target = _unique_file(_safe_target(target_dir, member.name))
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            try:
                with src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > limits.max_total_bytes:
                            dst.close()
                            target.unlink(missing_ok=True)
                            raise ExtractionError(
                                "total extracted size exceeds "
                                f"{limits.max_total_bytes} bytes"
                            )
                        dst.write(chunk)
            except ExtractionError:
                raise
            except (tarfile.TarError, OSError) as exc:
                target.unlink(missing_ok=True)
                raise ExtractionError(
                    f"failed extracting {member.name!r}: {exc}"
                ) from exc
            extracted.append(target)
    return extracted


def _extract_gzip(
    archive_path: Path, target_dir: Path, limits: ExtractionLimits,
) -> List[Path]:
    base = archive_path.name
    for ext in (".tar.gz", ".tgz", ".gz"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    if not base:
        base = archive_path.stem or "extracted"
    target = _unique_file(_safe_target(target_dir, base))
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with gzip.open(archive_path, "rb") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_single_file_bytes:
                    dst.close()
                    target.unlink(missing_ok=True)
                    raise ExtractionError(
                        f"gzip stream exceeds {limits.max_single_file_bytes} bytes"
                    )
                dst.write(chunk)
    except ExtractionError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        target.unlink(missing_ok=True)
        raise ExtractionError(f"gzip extraction failed: {exc}") from exc
    return [target]


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    limits: ExtractionLimits = DEFAULT_LIMITS,
    depth: int = 0,
) -> List[DetectionResult]:
    """
    Extract `archive_path` into a fresh subfolder of `dest_dir` and return
    DetectionResult entries for every extracted file. Nested archives are
    recursed into up to `limits.max_depth`. The original archive is never
    deleted or modified.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    if not archive_path.is_file():
        raise ExtractionError(f"not a file: {archive_path}")

    kind = _archive_kind(archive_path)
    if kind is None:
        raise ExtractionError(f"{archive_path.name!r} is not a recognised archive")
    if kind in ("7z", "rar"):
        raise ExtractionError(
            f"{kind!r} archive detected but no extractor is installed "
            f"(supported: {', '.join(sorted(PROCESSABLE_ARCHIVES))})"
        )

    target_dir = _unique_dest(dest_dir, archive_path.stem or "archive")

    if kind == "zip":
        files = _extract_zip(archive_path, target_dir, limits)
    elif kind == "tar":
        files = _extract_tar(archive_path, target_dir, limits)
    elif kind == "gzip":
        files = _extract_gzip(archive_path, target_dir, limits)
    else:
        raise ExtractionError(f"unhandled archive kind: {kind}")

    results: List[DetectionResult] = []
    for path in files:
        result = classify(path)
        if result.category == CATEGORY_ARCHIVE and depth < limits.max_depth:
            try:
                nested = extract_archive(path, target_dir, limits, depth + 1)
                results.extend(nested)
            except ExtractionError as exc:
                result.warnings.append(f"nested archive not expanded: {exc}")
                results.append(result)
        else:
            results.append(result)
    return results


def extract_many(
    archive_paths: List[Path],
    dest_dir: Path,
    limits: ExtractionLimits = DEFAULT_LIMITS,
) -> List[DetectionResult]:
    """Extract every archive in a list, converting failures into
    DetectionResult entries with warnings so the caller still has a
    record of every archive it was given."""
    out: List[DetectionResult] = []
    for path in archive_paths:
        try:
            out.extend(extract_archive(path, dest_dir, limits))
        except ExtractionError as exc:
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            out.append(DetectionResult(
                path=Path(path),
                filename=Path(path).name,
                size=size,
                extension=Path(path).suffix.lower(),
                extension_category=CATEGORY_ARCHIVE,
                header_category=CATEGORY_ARCHIVE,
                category=CATEGORY_ARCHIVE,
                role="archive",
                confidence=0.0,
                evidence=["archive extraction failed"],
                warnings=[str(exc)],
                metadata={},
            ))
    return out
