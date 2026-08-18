"""
Tests for image dimension resolution.

These matter more than they look. Wrong dimensions do not raise, do not 500, and
do not show up in any health check — they render the pre-annotation in the wrong
place on the canvas. The annotator then corrects a polygon that was never the
model's output, and the effort telemetry from that correction describes something
the policy never proposed. Every downstream number is then wrong in a way nothing
downstream can detect.

The header parsers are hand-rolled (no Pillow), so they are exercised against
real byte layouts rather than mocked.
"""
from __future__ import annotations

import struct

import pytest

from app.image_meta import ImageDimensionResolver, _dims_from_task_data, parse_dimensions


# ----------------------------------------------------------------------
# Header builders — minimal but structurally real files
# ----------------------------------------------------------------------

def png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\xf7\x00\x00"


def bmp(width: int, height: int) -> bytes:
    header = b"BM" + struct.pack("<IHHI", 70, 0, 0, 54)
    info = struct.pack("<Iii", 40, width, height)
    return header + info + b"\x01\x00\x18\x00"


def jpeg(width: int, height: int, *, with_app0: bool = True) -> bytes:
    out = b"\xff\xd8"
    if with_app0:
        payload = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        out += b"\xff\xe0" + struct.pack(">H", len(payload) + 2) + payload
    # SOF0: length, precision, height, width, components
    sof = b"\x08" + struct.pack(">HH", height, width) + b"\x03"
    out += b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
    return out


def webp_lossy(width: int, height: int) -> bytes:
    body = b"WEBP" + b"VP8 " + struct.pack("<I", 16)
    frame = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height)
    return b"RIFF" + struct.pack("<I", len(body) + len(frame)) + body + frame


# ----------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------

@pytest.mark.parametrize("builder", [png, gif, bmp, jpeg, webp_lossy])
def test_each_format_reports_its_dimensions(builder):
    assert parse_dimensions(builder(1920, 1080)) == (1920, 1080)


@pytest.mark.parametrize("builder", [png, gif, bmp, jpeg, webp_lossy])
def test_width_and_height_are_not_transposed(builder):
    """
    A transposed parser passes every square-image test and silently mirrors every
    mask on a non-square one, so the asymmetry is asserted explicitly.
    """
    assert parse_dimensions(builder(800, 600)) == (800, 600)


def test_jpeg_dimensions_are_read_from_sof_not_app0():
    """JPEG stores dimensions in a frame header that can sit behind other segments."""
    assert parse_dimensions(jpeg(1024, 768, with_app0=True)) == (1024, 768)
    assert parse_dimensions(jpeg(1024, 768, with_app0=False)) == (1024, 768)


def test_bmp_top_down_negative_height_is_read_as_positive():
    data = bmp(640, -480)
    assert parse_dimensions(data) == (640, 480)


def test_unrecognised_bytes_return_none_rather_than_guessing():
    assert parse_dimensions(b"this is not an image at all, not even close") is None


def test_truncated_header_returns_none():
    assert parse_dimensions(png(800, 600)[:12]) is None


def test_empty_input_returns_none():
    assert parse_dimensions(b"") is None


# ----------------------------------------------------------------------
# Resolution order
# ----------------------------------------------------------------------

def test_task_data_dimensions_win_over_probing(settings):
    resolver = ImageDimensionResolver(settings)
    dims = resolver.resolve("http://example.invalid/x.png", task_data={"width": 321, "height": 123})
    assert (dims.width, dims.height, dims.source) == (321, 123, "task_data")


@pytest.mark.parametrize(
    "data",
    [
        {"width": 800, "height": 600},
        {"image_width": 800, "image_height": 600},
        {"original_width": 800, "original_height": 600},
    ],
)
def test_task_data_dimension_key_aliases(data):
    assert _dims_from_task_data(data) == (800, 600)


@pytest.mark.parametrize(
    "data",
    [None, {}, {"width": 800}, {"width": 0, "height": 600}, {"width": "wide", "height": 600}],
)
def test_incomplete_or_nonsense_task_data_is_ignored(data):
    assert _dims_from_task_data(data) is None


def test_local_file_is_probed_from_disk(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    image = tmp_path / "frame.png"
    image.write_bytes(png(1280, 720))

    dims = ImageDimensionResolver(load_settings()).resolve(str(image))
    assert (dims.width, dims.height, dims.source) == (1280, 720, "probe")


def test_unreachable_image_falls_back_to_the_configured_default(settings, monkeypatch):
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    reloaded = load_settings()
    dims = ImageDimensionResolver(reloaded).resolve("file:///nowhere/missing.png")
    assert (dims.width, dims.height) == (reloaded.default_image_width, reloaded.default_image_height)
    assert dims.source == "fixed"


# ----------------------------------------------------------------------
# Reliability — a failed probe must not masquerade as a measurement
# ----------------------------------------------------------------------

def test_a_failed_probe_is_marked_unreliable(settings, monkeypatch):
    """
    Observed for real: probing images.cocodataset.org fails certificate
    verification, and a 640x480 image silently became the configured 1920x1080
    default. A 3x scale error renders the mask far off-target, the annotator
    corrects it anyway, and the telemetry describes a mask the policy never
    proposed. `/predict` refuses to serve on this flag.
    """
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    dims = ImageDimensionResolver(load_settings()).resolve("file:///nowhere/missing.png")
    assert dims.reliable is False


def test_explicitly_fixed_dimensions_are_reliable(settings):
    """`IMAGE_DIM_SOURCE=fixed` is an operator decision, not a guess."""
    dims = ImageDimensionResolver(settings).resolve("file:///nowhere/missing.png")
    assert dims.source == "fixed"
    assert dims.reliable is True


def test_successful_probe_is_reliable(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    image = tmp_path / "frame.png"
    image.write_bytes(png(320, 240))
    assert ImageDimensionResolver(load_settings()).resolve(str(image)).reliable is True


def test_task_data_dimensions_are_reliable(settings):
    dims = ImageDimensionResolver(settings).resolve("http://x/y.png", task_data={"width": 5, "height": 6})
    assert dims.reliable is True


# ----------------------------------------------------------------------
# Against a real file, not a synthetic header
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Locally served assets
# ----------------------------------------------------------------------

def test_own_asset_url_is_read_from_disk_not_fetched(settings, tmp_path, monkeypatch):
    """
    An image this service serves itself must not be fetched over HTTP from this
    service. Besides the pointless round trip, nothing is listening under a test
    client, so the probe would fail and the task would be refused.
    """
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path))
    from app.config import load_settings

    (tmp_path / "frame.png").write_bytes(png(640, 480))

    dims = ImageDimensionResolver(load_settings()).resolve(
        "http://localhost:8003/assets/frame.png"
    )
    assert (dims.width, dims.height, dims.reliable) == (640, 480, True)
    assert dims.source == "probe"


def test_asset_path_traversal_is_refused(settings, tmp_path, monkeypatch):
    """`image_url` arrives from another service, so the assets path is untrusted input."""
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path / "assets"))
    from app.config import load_settings

    (tmp_path / "assets").mkdir()
    (tmp_path / "secret.txt").write_bytes(b"token")

    resolver = ImageDimensionResolver(load_settings())
    assert resolver._local_asset("/assets/../secret.txt") is None


def test_non_asset_url_is_not_treated_as_local(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path))
    from app.config import load_settings

    assert ImageDimensionResolver(load_settings())._local_asset("/media/frame.png") is None


def test_real_jpeg_dimensions_are_read_correctly():
    """
    The parsers above are built from hand-written headers, which can agree with a
    buggy parser. This one is a real 640x480 JFIF photo from the COCO val2017 set.
    """
    from pathlib import Path

    image = Path(__file__).resolve().parents[3] / "tests" / "assets" / "test.jpg"
    if not image.exists():
        pytest.skip("tests/assets/test.jpg not present")

    assert image.read_bytes()[:2] == b"\xff\xd8"
    assert parse_dimensions(image.read_bytes()[:65536]) == (640, 480)


def test_unsupported_scheme_falls_back_rather_than_raising(settings, monkeypatch):
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    dims = ImageDimensionResolver(load_settings()).resolve("s3://bucket/key.png")
    assert dims.source == "fixed"


def test_probe_result_is_cached(settings, tmp_path, monkeypatch):
    """One image is opened once; /predict runs on every task open."""
    monkeypatch.setenv("IMAGE_DIM_SOURCE", "probe")
    from app.config import load_settings

    image = tmp_path / "frame.png"
    image.write_bytes(png(640, 480))
    resolver = ImageDimensionResolver(load_settings())

    assert resolver.resolve(str(image)).width == 640
    image.unlink()  # a second read from disk would now fail
    assert resolver.resolve(str(image)).width == 640
