"""
planning/plan_validator.py
Cross-checks a parsed EditPlan against the files actually present on
disk. Produces a ValidationReport; never raises for user-level
problems (missing files, unprocessable images), only for programming
errors. The caller decides whether to abort based on the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from media.detector import DetectionResult, classify
from media.formats import CATEGORY_IMAGE
from planning.plan_parser import EditPlan, PlanImage
from utils.error_handler import PipelineError, STAGE_VALIDATE


@dataclass
class ValidationIssue:
    kind: str                       # short code: missing_file, unprocessable, ...
    message: str
    severity: str = "error"         # "error" | "warning"
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


class _Index:
    """Case-sensitive first pass, case-insensitive fallback. Also exposes
    numeric-prefix matches so plans can reference '001.jpg' when the disk
    has '001_cover.jpg'."""

    def __init__(self, files: List[Path]) -> None:
        self.by_name: Dict[str, Path] = {}
        self.by_lower: Dict[str, List[Path]] = {}
        self.by_prefix: Dict[int, List[Path]] = {}
        for path in files:
            self.by_name.setdefault(path.name, path)
            self.by_lower.setdefault(path.name.lower(), []).append(path)
            prefix = _numeric_prefix(path.name)
            if prefix is not None:
                self.by_prefix.setdefault(prefix, []).append(path)
        for group in self.by_lower.values():
            group.sort(key=lambda p: p.name.lower())
        for group in self.by_prefix.values():
            group.sort(key=lambda p: p.name.lower())

    def resolve(self, name: str) -> Optional[Path]:
        if name in self.by_name:
            return self.by_name[name]
        lower = name.lower()
        if lower in self.by_lower:
            return self.by_lower[lower][0]
        prefix = _numeric_prefix(name)
        if prefix is not None and prefix in self.by_prefix:
            return self.by_prefix[prefix][0]
        return None

    def ambiguous(self, name: str) -> Optional[List[Path]]:
        lower = name.lower()
        matches = self.by_lower.get(lower, [])
        if len(matches) > 1:
            return matches
        prefix = _numeric_prefix(name)
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
    index = _Index(disk_files)

    kept = plan.ordered_kept()
    resolved: Dict[str, Path] = {}
    detections: Dict[str, DetectionResult] = {}
    referenced: Set[Path] = set()

    for img in kept:
        path = index.resolve(img.file)
        if path is None:
            issues.append(ValidationIssue(
                kind="missing_file",
                message=f"plan references '{img.file}' but it is not present",
                files=[img.file],
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
        resolved[img.file] = path

        detection = classify(path)
        detections[img.file] = detection

        if detection.category != CATEGORY_IMAGE:
            issues.append(ValidationIssue(
                kind="unprocessable",
                message=(
                    f"'{img.file}' resolved to '{path.name}' but was detected "
                    f"as {detection.category}, not an image"
                ),
                files=[img.file, path.name],
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
        path = resolved.get(img.file)
        if path is not None:
            reused.setdefault(path, []).append(img.file)
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

    kept_names_lower = {img.file.lower() for img in kept}
    for path in disk_files:
        if path in referenced:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".jpe", ".jfif",
                                        ".png", ".gif", ".webp", ".bmp",
                                        ".tif", ".tiff", ".avif", ".heic",
                                        ".heif", ".jxl", ".ico"}:
            continue
        if path.name.lower() in kept_names_lower:
            continue
        issues.append(ValidationIssue(
            kind="extra_image",
            message=f"'{path.name}' is on disk but not referenced by the plan",
            severity="warning",
            files=[path.name],
        ))

    rejected_files = [img.file for img in plan.rejected()]
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
    """Convenience: raise PipelineError if the report has errors."""
    report = validate_plan(plan, media_dir)
    if not report.ok:
        raise PipelineError(
            STAGE_VALIDATE,
            report.user_summary(),
            hint="fix the reported items in edit_plan.json or re-upload the missing files",
        )
    return report
