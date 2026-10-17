"""
Unit tests for planning/plan_parser.py and planning/plan_validator.py.

No network, no ffmpeg, no models. Uses only tmp_path for media fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.plan_parser import (
    EditPlan,
    PlanParseError,
    looks_like_edit_plan,
    parse_plan,
    parse_plan_file,
    parse_plan_text,
)
from planning.plan_validator import validate_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_plan_dict():
    return {
        "version": 1,
        "images": [
            {"file": "001.jpg", "story_line": "Kael entered the forest.",
             "keep": True, "order": 1},
            {"file": "002.jpg", "story_line": "A dragon appeared.",
             "keep": True, "order": 2, "effects": {"zoom": True}},
        ],
    }


def _write_image(path: Path, data: bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00") -> None:
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# plan_parser — valid inputs
# ---------------------------------------------------------------------------

def test_parse_valid_plan():
    plan = parse_plan(_good_plan_dict())
    assert plan.version == 1
    assert len(plan.images) == 2
    assert [i.file for i in plan.images] == ["001.jpg", "002.jpg"]
    assert plan.images[0].keep is True
    assert plan.images[1].effects == {"zoom": True}
    assert plan.warnings == []


def test_parse_plan_with_string_version_coerced():
    data = _good_plan_dict()
    data["version"] = "1"
    plan = parse_plan(data)
    assert plan.version == 1
    assert any("coerced" in w for w in plan.warnings)


def test_ordered_kept_uses_explicit_order_when_all_have_it():
    data = _good_plan_dict()
    data["images"][0]["order"] = 2
    data["images"][1]["order"] = 1
    plan = parse_plan(data)
    assert [i.file for i in plan.ordered_kept()] == ["002.jpg", "001.jpg"]


def test_ordered_kept_preserves_file_order_when_partial_order():
    data = _good_plan_dict()
    data["images"][0]["order"] = 5
    del data["images"][1]["order"]
    plan = parse_plan(data)
    assert [i.file for i in plan.ordered_kept()] == ["001.jpg", "002.jpg"]
    assert any("mixed" in w for w in plan.warnings)


def test_keep_false_is_skipped_in_ordered_kept():
    data = _good_plan_dict()
    data["images"][1]["keep"] = False
    plan = parse_plan(data)
    kept = plan.ordered_kept()
    rejected = plan.rejected()
    assert [i.file for i in kept] == ["001.jpg"]
    assert [i.file for i in rejected] == ["002.jpg"]


def test_keep_true_with_empty_story_line_warns_but_parses():
    data = _good_plan_dict()
    data["images"][0]["story_line"] = ""
    plan = parse_plan(data)
    assert any("story_line" in w for w in plan.warnings)


def test_unknown_top_level_and_image_keys_produce_warnings():
    data = _good_plan_dict()
    data["extra_top"] = "ignored"
    data["images"][0]["extra_image_key"] = 42
    plan = parse_plan(data)
    assert any("top-level" in w for w in plan.warnings)
    assert any("unknown keys" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# plan_parser — invalid inputs raise PlanParseError
# ---------------------------------------------------------------------------

def test_missing_version_raises():
    data = _good_plan_dict()
    del data["version"]
    with pytest.raises(PlanParseError, match="version"):
        parse_plan(data)


def test_unsupported_version_raises():
    data = _good_plan_dict()
    data["version"] = 99
    with pytest.raises(PlanParseError, match="unsupported"):
        parse_plan(data)


def test_root_not_object_raises():
    with pytest.raises(PlanParseError, match="object"):
        parse_plan([{"file": "001.jpg"}])


def test_images_not_array_raises():
    with pytest.raises(PlanParseError, match="array"):
        parse_plan({"version": 1, "images": "not a list"})


def test_empty_images_array_raises():
    with pytest.raises(PlanParseError, match="empty"):
        parse_plan({"version": 1, "images": []})


def test_missing_file_field_raises():
    with pytest.raises(PlanParseError, match="file"):
        parse_plan({"version": 1, "images": [{"keep": True}]})


def test_empty_file_string_raises():
    with pytest.raises(PlanParseError, match="file"):
        parse_plan({"version": 1, "images": [{"file": "   ", "keep": True}]})


def test_keep_not_boolean_raises():
    with pytest.raises(PlanParseError, match="keep"):
        parse_plan({"version": 1, "images": [
            {"file": "001.jpg", "keep": "yes"}
        ]})


def test_order_not_integer_raises():
    with pytest.raises(PlanParseError, match="order"):
        parse_plan({"version": 1, "images": [
            {"file": "001.jpg", "keep": True, "order": "1"}
        ]})


def test_effects_not_object_raises():
    with pytest.raises(PlanParseError, match="effects"):
        parse_plan({"version": 1, "images": [
            {"file": "001.jpg", "keep": True, "effects": "zoom"}
        ]})


def test_all_images_keep_false_raises():
    with pytest.raises(PlanParseError, match="keep=false"):
        parse_plan({"version": 1, "images": [
            {"file": "001.jpg", "keep": False},
            {"file": "002.jpg", "keep": False},
        ]})


def test_duplicate_file_and_order_warn_not_raise():
    plan = parse_plan({
        "version": 1,
        "images": [
            {"file": "001.jpg", "keep": True, "order": 1},
            {"file": "001.jpg", "keep": True, "order": 1},
        ],
    })
    joined = " ".join(plan.warnings)
    assert "duplicate 'file'" in joined
    assert "duplicate 'order'" in joined


# ---------------------------------------------------------------------------
# parse_plan_text / parse_plan_file
# ---------------------------------------------------------------------------

def test_parse_plan_text_invalid_json_raises():
    with pytest.raises(PlanParseError, match="not valid JSON"):
        parse_plan_text("{ not json")


def test_parse_plan_file_roundtrip(tmp_path: Path):
    path = tmp_path / "edit_plan.json"
    path.write_text(json.dumps(_good_plan_dict()), encoding="utf-8")
    plan = parse_plan_file(path)
    assert plan.version == 1
    assert len(plan.images) == 2


def test_parse_plan_file_missing(tmp_path: Path):
    with pytest.raises(PlanParseError, match="cannot read"):
        parse_plan_file(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# looks_like_edit_plan sniff
# ---------------------------------------------------------------------------

def test_looks_like_edit_plan_true():
    assert looks_like_edit_plan({
        "version": 1,
        "images": [{"file": "a.jpg", "keep": True}],
    })


def test_looks_like_edit_plan_false_for_legacy_timeline():
    assert not looks_like_edit_plan([{"image": "a.jpg", "start": 0, "end": 1}])


def test_looks_like_edit_plan_false_when_no_keep():
    assert not looks_like_edit_plan({
        "version": 1, "images": [{"file": "a.jpg"}],
    })


def test_looks_like_edit_plan_false_when_images_empty():
    assert not looks_like_edit_plan({"version": 1, "images": []})


# ---------------------------------------------------------------------------
# plan_validator
# ---------------------------------------------------------------------------

def test_validate_ok_when_all_files_present(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.jpg")
    _write_image(media / "002.jpg")

    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    assert report.ok
    assert len(report.kept_images) == 2
    assert report.errors() == []


def test_validate_missing_file_is_error(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.jpg")
    # 002.jpg missing

    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    assert not report.ok
    kinds = [i.kind for i in report.errors()]
    assert "missing_file" in kinds


def test_validate_extra_image_on_disk_is_warning(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("001.jpg", "002.jpg", "999_extra.jpg"):
        _write_image(media / name)

    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    assert report.ok
    assert any(i.kind == "extra_image" for i in report.warnings())


def test_validate_non_image_reference_is_error(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.jpg")
    (media / "002.jpg").write_bytes(b"this is plain text, not an image")

    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    assert not report.ok
    assert any(i.kind == "unprocessable" for i in report.errors())


def test_validate_numeric_prefix_fallback(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001_cover.jpg")
    _write_image(media / "002_cover.jpg")

    plan = parse_plan({
        "version": 1,
        "images": [
            {"file": "001.jpg", "keep": True, "story_line": "a"},
            {"file": "002.jpg", "keep": True, "story_line": "b"},
        ],
    })
    report = validate_plan(plan, media)
    assert report.ok


def test_validate_case_insensitive_fallback(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.JPG")
    _write_image(media / "002.JPG")

    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    assert report.ok


def test_validate_duplicate_plan_references_warn(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.jpg")

    plan = parse_plan({
        "version": 1,
        "images": [
            {"file": "001.jpg", "keep": True, "order": 1, "story_line": "a"},
            {"file": "001.jpg", "keep": True, "order": 2, "story_line": "b"},
        ],
    })
    report = validate_plan(plan, media)
    assert report.ok
    assert any(i.kind == "duplicate_image_use" for i in report.warnings())


def test_validate_rejected_images_surface_as_warning(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_image(media / "001.jpg")
    _write_image(media / "002.jpg")

    data = _good_plan_dict()
    data["images"][1]["keep"] = False
    plan = parse_plan(data)
    report = validate_plan(plan, media)
    assert report.ok
    assert any(i.kind == "rejected_images" for i in report.warnings())


def test_validate_missing_media_dir_raises(tmp_path: Path):
    plan = parse_plan(_good_plan_dict())
    with pytest.raises(Exception):
        validate_plan(plan, tmp_path / "does_not_exist")


def test_validate_user_summary_mentions_errors(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    plan = parse_plan(_good_plan_dict())
    report = validate_plan(plan, media)
    summary = report.user_summary()
    assert "missing_file" in summary


# ---------------------------------------------------------------------------
# EditPlan wrapper sanity
# ---------------------------------------------------------------------------

def test_editplan_kept_and_rejected_are_disjoint():
    data = _good_plan_dict()
    data["images"][1]["keep"] = False
    plan = parse_plan(data)
    kept = {i.file for i in plan.kept()}
    rejected = {i.file for i in plan.rejected()}
    assert kept.isdisjoint(rejected)
    assert kept | rejected == {i.file for i in plan.images}
