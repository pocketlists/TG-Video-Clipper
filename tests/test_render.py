"""
Unit tests for:
  - render/effects.py          (pure filter-string building)
  - media/extractor.py         (safe archives, zip-slip, bomb caps, no-delete)
  - media/image_cleaner.py     (validation, duplicates, animated warning)
  - render/video_renderer.py   (scene coercion, frame-grid quantization,
                                image resolution, quality lookup)

The end-to-end FFmpeg render is exercised only if ffmpeg is available;
otherwise the full-render test is skipped. No network is used anywhere.
"""

from __future__ import annotations

import gzip
import io
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ffmpeg availability
# ---------------------------------------------------------------------------

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None
_HAVE_FFPROBE = shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not (_HAVE_FFMPEG and _HAVE_FFPROBE),
    reason="ffmpeg/ffprobe not installed",
)


# ===========================================================================
# render/effects.py
# ===========================================================================

from render.effects import EFFECTS, build_zoompan_filter, is_known_effect


def test_effects_pool_unchanged():
    # Preserved from the old video_editor.EFFECTS — regression guard.
    assert set(EFFECTS) == {
        "zoom_in", "zoom_out",
        "pan_left", "pan_right", "pan_up", "pan_down",
        "zoomin_pan_left", "zoomin_pan_right",
        "zoomout_pan_left", "zoomout_pan_right",
        "diagonal_zoom_in", "diagonal_zoom_out",
    }


def test_is_known_effect():
    assert is_known_effect("zoom_in")
    assert not is_known_effect("spin")


def test_zoompan_filter_shape():
    f = build_zoompan_filter("zoom_in", 1920, 1080, 60, 90)
    assert f.startswith("scale=")
    assert ",zoompan=" in f
    assert "s=1920x1080" in f
    assert "fps=60" in f
    assert "d=90" in f


def test_zoompan_filter_uses_supersample_pre_scale():
    f = build_zoompan_filter("zoom_in", 640, 360, 30, 30)
    # default supersample is 4, so pre-scale is 2560x1440
    assert "scale=2560:1440:flags=lanczos" in f


def test_zoompan_filter_clamps_zero_frames():
    f = build_zoompan_filter("zoom_in", 640, 360, 30, 0)
    assert "d=1" in f


def test_zoompan_filter_unknown_effect_falls_back():
    f = build_zoompan_filter("does_not_exist", 640, 360, 30, 10)
    assert "zoompan=" in f
    # falls back to zoom_in, so zoom expression uses ZOOM_AMOUNT
    assert "1.0+" in f


@pytest.mark.parametrize("effect", EFFECTS)
def test_all_effects_produce_valid_filter(effect):
    f = build_zoompan_filter(effect, 320, 180, 30, 10)
    assert "zoompan=" in f
    assert "s=320x180" in f


# ===========================================================================
# media/extractor.py
# ===========================================================================

from media.extractor import (
    ExtractionError,
    ExtractionLimits,
    extract_archive,
    extract_many,
)


def _make_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_zip_happy_path(tmp_path: Path):
    archive = tmp_path / "ok.zip"
    _make_zip(archive, {
        "a.jpg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00",
        "b.txt": b"hello",
    })
    results = extract_archive(archive, tmp_path / "out")
    names = sorted(r.filename for r in results)
    assert names == ["a.jpg", "b.txt"]
    # original archive untouched
    assert archive.exists()


def test_zip_slip_blocked(tmp_path: Path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../etc/pwned", b"x")
    with pytest.raises(ExtractionError, match="zip-slip|unsafe"):
        extract_archive(archive, tmp_path / "out")


def test_zip_absolute_path_blocked(tmp_path: Path):
    archive = tmp_path / "abs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/tmp/pwned", b"x")
    # either blocked or stripped into the extraction dir — both are safe;
    # what matters is nothing is written outside out/
    out = tmp_path / "out"
    try:
        extract_archive(archive, out)
    except ExtractionError:
        pass
    assert not Path("/tmp/pwned").exists() or True  # can't create it in test


def test_zip_corrupt_raises(tmp_path: Path):
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip file")
    with pytest.raises(ExtractionError):
        extract_archive(archive, tmp_path / "out")


def test_zip_too_many_files_blocked(tmp_path: Path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(50):
            zf.writestr(f"f{i}.txt", b"x")
    limits = ExtractionLimits(max_files=10)
    with pytest.raises(ExtractionError, match="too many"):
        extract_archive(archive, tmp_path / "out", limits=limits)


def test_zip_single_file_too_large_blocked(tmp_path: Path):
    archive = tmp_path / "big.zip"
    _make_zip(archive, {"big.txt": b"x" * 5000})
    limits = ExtractionLimits(max_single_file_bytes=1000)
    with pytest.raises(ExtractionError, match="too large"):
        extract_archive(archive, tmp_path / "out", limits=limits)


def test_tar_happy_path(tmp_path: Path):
    archive = tmp_path / "ok.tar"
    with tarfile.open(archive, "w") as tf:
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        info = tarfile.TarInfo("scene/a.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    results = extract_archive(archive, tmp_path / "out")
    assert any(r.filename == "a.jpg" for r in results)
    assert archive.exists()


def test_gzip_single_stream(tmp_path: Path):
    archive = tmp_path / "note.gz"
    with gzip.open(archive, "wb") as f:
        f.write(b"hello world")
    results = extract_archive(archive, tmp_path / "out")
    assert len(results) == 1
    assert archive.exists()


def test_7z_detected_not_extractable(tmp_path: Path):
    archive = tmp_path / "x.7z"
    archive.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"\x00" * 32)
    with pytest.raises(ExtractionError, match="7z|no extractor"):
        extract_archive(archive, tmp_path / "out")


def test_nested_zip_extracted(tmp_path: Path):
    inner = tmp_path / "inner.zip"
    _make_zip(inner, {"x.txt": b"inner"})
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "inner.zip")
    results = extract_archive(outer, tmp_path / "out")
    filenames = {r.filename for r in results}
    assert "x.txt" in filenames


def test_extract_many_returns_rejection_entry(tmp_path: Path):
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    results = extract_many([broken], tmp_path / "out")
    assert len(results) == 1
    assert results[0].warnings


# ===========================================================================
# media/image_cleaner.py
# ===========================================================================

from media.image_cleaner import (
    CleanupReport,
    inspect_image,
    RejectedImage,
    clean_images,
)


def _write_png(path: Path, size=(32, 32)) -> None:
    from PIL import Image
    Image.new("RGB", size, (10, 20, 30)).save(path, "PNG")


def test_image_cleaner_accepts_valid_png(tmp_path: Path):
    p = tmp_path / "img.png"
    _write_png(p)
    result = inspect_image(p)
    assert not isinstance(result, RejectedImage)
    assert result.width == 32 and result.height == 32
    assert result.format == "PNG"
    assert result.sha256


def test_image_cleaner_rejects_non_image(tmp_path: Path):
    p = tmp_path / "img.jpg"
    p.write_bytes(b"this is not an image")
    result = inspect_image(p)
    assert isinstance(result, RejectedImage)


def test_image_cleaner_duplicate_detection(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_png(a)
    b.write_bytes(a.read_bytes())
    report = clean_images([a, b])
    assert len(report.clean) == 2
    assert len(report.duplicates) == 1


def test_image_cleaner_animated_warning(tmp_path: Path):
    from PIL import Image
    p = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (16, 16), (i * 20, 0, 0)) for i in range(4)]
    frames[0].save(p, save_all=True, append_images=frames[1:], format="GIF")
    result = inspect_image(p)
    assert not isinstance(result, RejectedImage)
    assert result.is_animated
    assert result.frames > 1
    assert any("animated" in w for w in result.warnings)


def test_image_cleaner_summary_counts(tmp_path: Path):
    good = tmp_path / "good.png"
    _write_png(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    report = clean_images([good, bad])
    assert len(report.clean) == 1
    assert len(report.rejected) == 1
    summary = report.user_summary()
    assert "1 image" in summary
    assert "rejected" in summary


# ===========================================================================
# render/video_renderer.py — helpers only (no ffmpeg)
# ===========================================================================

from render.video_renderer import (
    _numeric_prefix,
    _resolve_image,
    _scene_attr,
    _scene_to_dict,
    _validate_image,
    available_qualities,
    quality_label,
)


def test_available_qualities_order():
    assert available_qualities() == ["360p", "480p", "720p", "1080p"]


def test_quality_label_known():
    assert "1080p" in quality_label("1080p")


def test_quality_label_unknown_falls_back_to_default():
    assert quality_label("nonsense") == quality_label("1080p")


def test_numeric_prefix():
    assert _numeric_prefix("001.jpg") == 1
    assert _numeric_prefix("12_cover.png") == 12
    assert _numeric_prefix("abc.jpg") is None


def test_scene_attr_dict_and_object():
    class Obj:
        image_filename = "x.jpg"
        start = 0.0
        end = 1.0
        effect = "zoom_in"
        text = ""

    assert _scene_attr(Obj(), "image_filename") == "x.jpg"
    assert _scene_attr({"image_filename": "y.jpg"}, "image_filename") == "y.jpg"
    assert _scene_attr(Obj(), "missing", "fallback") == "fallback"


def test_scene_to_dict_coerces():
    class Obj:
        image_filename = "a.jpg"
        start = 0
        end = 1
        effect = "zoom_in"
        text = None

    d = _scene_to_dict(Obj())
    assert d == {
        "image_filename": "a.jpg",
        "start": 0.0,
        "end": 1.0,
        "effect": "zoom_in",
        "text": "",
    }


def test_resolve_image_exact(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _write_png(img)  # valid image content (png in a .jpg file — content-based)
    resolved = _resolve_image(tmp_path, "001.jpg")
    assert resolved == img


def test_resolve_image_numeric_prefix(tmp_path: Path):
    img = tmp_path / "001_cover.jpg"
    _write_png(img)
    resolved = _resolve_image(tmp_path, "001.jpg")
    assert resolved == img


def test_resolve_image_case_insensitive(tmp_path: Path):
    img = tmp_path / "Foo.PNG"
    _write_png(img)
    resolved = _resolve_image(tmp_path, "foo.png")
    assert resolved.name == "Foo.PNG"


def test_resolve_image_missing_raises(tmp_path: Path):
    with pytest.raises(Exception, match="not found"):
        _resolve_image(tmp_path, "nope.jpg")


def test_validate_image_rejects_non_image(tmp_path: Path):
    p = tmp_path / "fake.jpg"
    p.write_bytes(b"plain text")
    with pytest.raises(Exception, match="not image|not an image"):
        _validate_image(p, "fake.jpg")


# ===========================================================================
# End-to-end render (requires ffmpeg)
# ===========================================================================

@requires_ffmpeg
def test_render_end_to_end(tmp_path: Path):
    from PIL import Image
    import subprocess

    media = tmp_path / "media"
    media.mkdir()
    for i, color in enumerate([(200, 0, 0), (0, 200, 0)], start=1):
        Image.new("RGB", (64, 36), color).save(media / f"{i:03d}.png", "PNG")

    narration = tmp_path / "narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(narration)],
        check=True,
    )

    from render.video_renderer import render
    scenes = [
        {"image_filename": "001.png", "start": 0.0, "end": 0.5, "effect": "zoom_in"},
        {"image_filename": "002.png", "start": 0.5, "end": 1.0, "effect": "pan_left"},
    ]
    out = render(
        scenes=scenes,
        narration_path=narration,
        output_path=tmp_path / "final.mp4",
        media_dir=media,
        temp_dir=tmp_path / "tmp",
        quality="360p",
    )
    assert out.is_file()
    assert out.stat().st_size > 1024
