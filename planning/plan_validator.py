"""
planning/plan_validator.py

Cross-checks a parsed EditPlan against the files actually present on
disk. Produces a ValidationReport; never raises for user-level
problems (missing files, unprocessable images), only for programming
errors. The caller decides whether to abort based on the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from media.detector import DetectionResult, classify
from media.formats import CATEGORY_IMAGE
from planning.plan_parser import EditPlan, PlanImage
from utils.error_handler import PipelineError, STAGE_VALIDATE


@dataclass
class ValidationIssue:
    kind: str
    message: str
    severity: str = "error"
    files: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    kept_images: List[PlanImage] = field(default_factory=list)
    resolved_paths: Dict[str, Path] = field(default_factory=dict)
    detections: Dict[str, DetectionResult] = field(default_factory=dict)
    extra_files: List[str] = field(default_factory=list)

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def user_summary(self) -> str:
        if self.ok and not self.warnings():
            return f"✅ Plan validated: {len(self.kept_images)} images ready."
        lines: List[str] = []
        errs = self.errors()
        warns = self.warnings()
        if errs:
            lines.append(f"❌ {len(errs)} problem(s) — render cannot start:")
            for issue in errs[:20]:
                lines.append(f"  • [{issue.kind}] {issue.message}")
            if len(errs) > 20:
                lines.append(f"  … and {len(errs) - 20} more")
        if warns:
            lines.append(f"⚠️ {len(warns)} warning(s):")
            for issue in warns[:20]:
                lines.append(f"  • [{issue.kind}] {issue.message}")
            if len(warns) > 20:
                lines.append(f"  … and {len(warns) - 20} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter-number parsing + plan->disk archive remapping
#
# Problem: har chapter ke zip me SAME file naam hote hain (002.webp har
# chapter me hai). Sahi image sirf zip/chapter folder se pehchani jati hai.
#
# Rules (user ki requirement):
#   1. edit_plan.json me zip ka FULL naam hai aur wahi zip disk par
#      upload hua hai  ->  bilkul waisa hi rakho, kuch change mat karo.
#   2. edit_plan.json me full naam NAHI hai (zip field missing, ya
#      chhota/alag naam jaise "Ch1.zip")  ->  chapter number se match
#      karo. Uploaded zip ka naam kisi bhi format me ho sakta hai:
#      "Monster Streaming Star_Ch1.zip", "Monster Streaming
#      Star — Chapter 2.zip", "Chapter 3.zip", "ch-4.zip",
#      "series_part_2.zip" ... — naam se chapter number nikalo aur us
#      chapter ki plan entries ko wahi folder do.
#   3. Zip kis order me upload hue, isse koi farak nahi padta —
#      matching hamesha chapter number se hoti hai.
# ---------------------------------------------------------------------------

_CHAPTER_PATTERNS = (
    re.compile(r"chapter[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"(?:^|[^a-z0-9])ch[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"episode[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"(?:^|[^a-z0-9])ep[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"part[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"vol(?:ume)?[\s._-]*(\d{1,4})", re.IGNORECASE),
    re.compile(r"[\s._-](\d{1,4})$"),
)


def chapter_number_from_name(name: Optional[str]) -> Optional[int]:
    """Kisi bhi file/folder naam se chapter number nikalo, warna None.

    Sirf naam (extension ke bina bhi chalega):
        "Monster Streaming Star_Ch1"   -> 1
        "Monster Streaming Star — Chapter 12" -> 12
        "Chapter 3" / "ch-4" / "series_part_2" / "Story 7" -> number
    """
    if not name:
        return None
    stem = Path(str(name)).stem.strip() or str(name).strip()
    for pattern in _CHAPTER_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def remap_plan_archives(plan: EditPlan, media_dir: Path) -> List[str]:
    """Plan ki zip->folder mapping ko disk ke asal folders se milao.

    edit_plan.json FILE change nahi hoti — sirf memory me parsed plan
    theek hota hai. Return: kiye gaye remap ki note-lines (logging ke
    liye). Rule 1 (full naam + disk par maujud) wali entries ko hath
    bhi nahi lagaya jata.
    """
    media_dir = Path(media_dir)
    notes: List[str] = []
    if not media_dir.is_dir():
        return notes

    # Extractor har zip ko input/<zip-ke-naam-wala-folder>/ me kholta hai.
    top_dirs = sorted(
        (d for d in media_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name.lower(),
    )
    if not top_dirs:
        return notes  # loose images — koi zip folder hi nahi

    # Rule 1 check ke liye: disk par kisi bhi level par maujud folder naam.
    disk_dir_names = {
        d.name.lower()
        for d in media_dir.rglob("*")
        if d.is_dir()
    }

    chapter_dirs: Dict[int, Path] = {}
    unnumbered_dirs: List[Path] = []
    for folder in top_dirs:
        num = chapter_number_from_name(folder.name)
        if num is None:
            unnumbered_dirs.append(folder)
        elif num not in chapter_dirs:
            chapter_dirs[num] = folder

    changed: Dict[Tuple[Optional[str], str, Optional[int]], int] = {}
    for img in plan.images:
        stem = Path(img.zip).stem.strip() if img.zip else None
        if stem and stem.lower() in disk_dir_names:
            continue  # Rule 1: full naam hai aur disk par hai — as-is

        chapter = img.chapter
        if chapter is None:
            chapter = chapter_number_from_name(img.zip)

        target: Optional[Path] = None
        if chapter is not None:
            target = chapter_dirs.get(chapter)
            if (
                target is None
                and chapter == 1
                and len(top_dirs) == 1
                and len(unnumbered_dirs) == 1
            ):
                # Ek hi zip aaya hai aur uske naam me number nahi hai —
                # use chapter 1 se shuru mano.
                target = unnumbered_dirs[0]

        if target is None:
            continue

        new_zip = f"{target.name}.zip"
        if img.zip != new_zip:
            key = (img.zip, new_zip, chapter)
            changed[key] = changed.get(key, 0) + 1
            img.zip = new_zip
        if img.chapter is None and chapter is not None:
            img.chapter = chapter

    for (old_zip, new_zip, chapter), count in sorted(
        changed.items(), key=lambda kv: (kv[0][2] or 0, kv[0][1])
    ):
        notes.append(
            f"zip {old_zip!r} -> {new_zip!r} "
            f"({count} entries, chapter {chapter})"
        )
    return notes


class _Index:
    """
    Resolves image paths considering subdirectories (from zips/chapters),
    exact names, case-insensitive names, and numeric prefixes.
    """

    def __init__(self, files: List[Path], base_dir: Path) -> None:
        self.by_relpath: Dict[str, Path] = {}
        self.by_name: Dict[str, Path] = {}
        self.by_lower: Dict[str, List[Path]] = {}
        self.by_prefix: Dict[int, List[Path]] = {}

        for path in files:
            try:
                rel = path.relative_to(base_dir).as_posix()
                self.by_relpath[rel] = path
                self.by_relpath[rel.lower()] = path
            except ValueError:
                pass

            self.by_name.setdefault(path.name, path)
            self.by_lower.setdefault(path.name.lower(), []).append(path)
            prefix = _numeric_prefix(path.name)
            if prefix is not None:
                self.by_prefix.setdefault(prefix, []).append(path)

        for group in self.by_lower.values():
            group.sort(key=lambda p: p.name.lower())
        for group in self.by_prefix.values():
            group.sort(key=lambda p: p.name.lower())

    def resolve(self, name: str, *, folder_strict: bool = False) -> Optional[Path]:
        name_clean = name.replace("\\", "/").lstrip("/")

        # 1. Exact relative path
        if name_clean in self.by_relpath:
            return self.by_relpath[name_clean]

        # 2. Case-insensitive relative path
        lower_rel = name_clean.lower()
        if lower_rel in self.by_relpath:
            return self.by_relpath[lower_rel]

        basename = Path(name_clean).name

        # 2b. Folder-aware match: naam me folder diya hai to usi folder
        # (ya usi naam ke parent folder) wali file hi mile — dusre
        # chapter ki same-naam wali file kabhi nahi. Ek extra wrapper
        # folder (extraction stem) ke andar wala path bhi chalega.
        if "/" in name_clean:
            folder_part = Path(name_clean).parent.name.lower()
            suffix = "/" + lower_rel
            candidates = [
                p for rel, p in self.by_relpath.items()
                if rel == rel.lower() and rel.endswith(suffix)
            ]
            if candidates:
                return sorted(candidates, key=lambda p: str(p).lower())[0]
            same_folder = [
                p for p in self.by_lower.get(basename.lower(), [])
                if p.parent.name.lower() == folder_part
            ]
            if same_folder:
                return sorted(same_folder, key=lambda p: str(p).lower())[0]
            if folder_strict:
                # Folder diya tha lekin us folder me file nahi mili.
                # Dusre chapter se utha kar galat image lagane se
                # behtar hai: missing report karo.
                return None

        # 3. Exact filename
        if basename in self.by_name:
            if folder_strict and len(self.by_lower.get(basename.lower(), [])) > 1:
                return None
            return self.by_name[basename]

        # 4. Case-insensitive filename
        basename_lower = basename.lower()
        if basename_lower in self.by_lower:
            if folder_strict and len(self.by_lower[basename_lower]) > 1:
                return None
            return self.by_lower[basename_lower][0]

        # 5. Numeric prefix fallback
        prefix = _numeric_prefix(basename)
        if prefix is not None and prefix in self.by_prefix:
            if folder_strict and len(self.by_prefix[prefix]) > 1:
                return None
            return self.by_prefix[prefix][0]

        return None

    def ambiguous(self, name: str) -> Optional[List[Path]]:
        basename_lower = Path(name.replace("\\", "/")).name.lower()
        matches = self.by_lower.get(basename_lower, [])
        if len(matches) > 1:
            return matches
        prefix = _numeric_prefix(Path(name.replace("\\", "/")).name)
        if prefix is not None:
            matches = self.by_prefix.get(prefix, [])
            if len(matches) > 1:
                return matches
        return None


def _numeric_prefix(name: str) -> Optional[int]:
    i = 0
    while i < len(name) and name[i].isdigit():
        i += 1
    if i == 0:
        return None
    try:
        return int(name[:i])
    except ValueError:
        return None


def validate_plan(plan: EditPlan, media_dir: Path) -> ValidationReport:
    media_dir = Path(media_dir)
    issues: List[ValidationIssue] = []

    if not media_dir.is_dir():
        raise PipelineError(
            STAGE_VALIDATE,
            f"media directory does not exist: {media_dir}",
        )

    disk_files = sorted(
        (p for p in media_dir.rglob("*") if p.is_file()),
        key=lambda p: p.name.lower(),
    )
    index = _Index(disk_files, media_dir)

    kept = plan.ordered_kept()
    resolved: Dict[str, Path] = {}
    detections: Dict[str, DetectionResult] = {}
    referenced: Set[Path] = set()

    for img in kept:
        target_name = img.qualified_file
        # qualified_file me folder (zip/chapter) hai to STRICT resolve:
        # bare file-naam se dusre chapter ki same-naam file kabhi match
        # nahi hogi. Bina folder wale plan par purana fallback chalega.
        has_folder = "/" in target_name.replace("\\", "/")
        path = index.resolve(target_name, folder_strict=has_folder)
        if path is None and not has_folder:
            path = index.resolve(img.file)

        if path is None:
            issues.append(ValidationIssue(
                kind="missing_file",
                message=f"plan references '{target_name}' but it is not present",
                files=[target_name],
            ))
            continue

        ambiguous = index.ambiguous(img.file)
        if ambiguous is not None:
            issues.append(ValidationIssue(
                kind="ambiguous_reference",
                message=(
                    f"plan references '{img.file}' but multiple files match: "
                    f"{[p.name for p in ambiguous[:5]]}"
                ),
                severity="warning",
                files=[img.file],
            ))

        referenced.add(path)
        resolved[img.qualified_file] = path

        detection = classify(path)
        detections[img.qualified_file] = detection

        if detection.category != CATEGORY_IMAGE:
            issues.append(ValidationIssue(
                kind="unprocessable",
                message=(
                    f"'{target_name}' resolved to '{path.name}' but was detected "
                    f"as {detection.category}, not an image"
                ),
                files=[target_name, path.name],
            ))
            continue

        if detection.confidence < 0.65:
            issues.append(ValidationIssue(
                kind="low_confidence_image",
                message=(
                    f"'{path.name}' passed as image but detection was only "
                    f"{detection.confidence:.2f} confident"
                ),
                severity="warning",
                files=[path.name],
            ))

    reused: Dict[Path, List[str]] = {}
    for img in kept:
        path = resolved.get(img.qualified_file)
        if path is not None:
            reused.setdefault(path, []).append(img.qualified_file)
            
    for path, names in reused.items():
        if len(names) > 1:
            issues.append(ValidationIssue(
                kind="duplicate_image_use",
                message=(
                    f"image '{path.name}' is referenced by {len(names)} plan "
                    f"entries: {names}"
                ),
                severity="warning",
                files=names,
            ))

    kept_names_lower = {img.qualified_file.lower() for img in kept}
    kept_basenames_lower = {Path(img.file).name.lower() for img in kept}
    
    for path in disk_files:
        if path in referenced:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".jpe", ".jfif",
                                        ".png", ".gif", ".webp", ".bmp",
                                        ".tif", ".tiff", ".avif", ".heic",
                                        ".heif", ".jxl", ".ico"}:
            continue
            
        try:
            rel = path.relative_to(media_dir).as_posix().lower()
        except ValueError:
            rel = path.name.lower()
            
        if rel in kept_names_lower or path.name.lower() in kept_basenames_lower:
            continue
            
        issues.append(ValidationIssue(
            kind="extra_image",
            message=f"'{path.name}' is on disk but not referenced by the plan",
            severity="warning",
            files=[path.name],
        ))

    rejected_files = [img.qualified_file for img in plan.rejected()]
    if rejected_files:
        issues.append(ValidationIssue(
            kind="rejected_images",
            message=f"{len(rejected_files)} image(s) marked keep=false and will be skipped",
            severity="warning",
            files=rejected_files[:10],
        ))

    orders = [img.order for img in kept if img.order is not None]
    if orders and len(set(orders)) != len(orders):
        dupes = sorted({o for o in orders if orders.count(o) > 1})
        issues.append(ValidationIssue(
            kind="duplicate_order",
            message=f"duplicate 'order' values among kept images: {dupes}",
            severity="warning",
        ))

    has_error = any(i.severity == "error" for i in issues)
    return ValidationReport(
        ok=not has_error,
        issues=issues,
        kept_images=kept,
        resolved_paths=resolved,
        detections=detections,
        extra_files=[p.name for p in disk_files
                     if p not in referenced and p.suffix.lower() in kept_suffixes(kept)],
    )


def kept_suffixes(_: List[PlanImage]) -> Set[str]:
    return {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".webp", ".bmp",
            ".tif", ".tiff", ".avif", ".heic", ".heif", ".jxl", ".ico"}


def assert_valid(plan: EditPlan, media_dir: Path) -> ValidationReport:
    report = validate_plan(plan, media_dir)
    if not report.ok:
        raise PipelineError(
            STAGE_VALIDATE,
            report.user_summary(),
            hint="fix the reported items in edit_plan.json or re-upload the missing files",
        )
    return report
