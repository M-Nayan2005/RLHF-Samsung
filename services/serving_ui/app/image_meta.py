"""
Resolving an image's pixel dimensions.

Why this module has to exist: `PolygonMask.points` are **absolute pixels**
(`tier1_ingestion.py` says so explicitly), but Label Studio stores polygon
coordinates as **percentages of the image, 0-100** - visible in Dev 4's own
fixture, whose region points are values like `[10.1, 12.4]`. Converting between
the two needs the image's width and height, and **no frozen contract carries
them**. `QueueTask` has `image_url` and a pixel-space `bounding_box`, and that
is all.

Recorded as divergence D12. It is a genuine hole in the contracts rather than a
deferral, but it is fixable entirely inside this service, so it does not need a
schema change: read the dimensions from the image itself.

Resolution order, most to least trustworthy:

1. `width` / `height` on the Label Studio `task.data` - if whoever imported the
   task already knew, believe them.
2. Probe the image: fetch just enough bytes to read the header.
3. `DEFAULT_IMAGE_WIDTH` / `DEFAULT_IMAGE_HEIGHT`, logged as a warning. Wrong
   dimensions do not fail loudly - they render the mask in the wrong place and
   the annotator silently corrects a polygon that was never the model's output,
   which poisons the reward. So the fallback is a last resort and it says so.

Header parsing is done by hand rather than with Pillow. The formats below need
about eighty lines between them, all of it reading fixed-offset integers, and
that is a better trade than adding an image library to a service that never
touches pixels.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx

from .config import Settings

log = logging.getLogger(__name__)

# Enough for a PNG/GIF/BMP/WEBP header and, in practice, a JPEG's first SOF.
_HEADER_BYTES = 65536


class ImageDims(NamedTuple):
    width: int
    height: int
    source: str  # task_data | probe | fixed
    reliable: bool = True
    """
    False when these dimensions are a guess rather than a measurement.

    The caller must not serve a pre-annotation built on unreliable dimensions.
    Getting them wrong does not fail - it renders the mask at the wrong scale and
    in the wrong place, the annotator "corrects" a polygon the model never
    proposed, and the resulting effort telemetry is not merely missing but
    actively misleading. An empty prediction loses one rollout; a wrong-scale one
    corrupts the reward signal with a plausible-looking number.

    Observed for real: probing https://images.cocodataset.org/... fails TLS
    verification (the host's certificate does not match), and the 640x480 image
    silently became the configured 1920x1080 default - a 3x scale error with a
    warning in the log and nothing else to show for it.
    """


# --------------------------------------------------------------------------
# Header parsers
# --------------------------------------------------------------------------

def parse_dimensions(head: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from an image file's leading bytes, or None if unrecognised."""
    for parser in (_png, _gif, _bmp, _webp, _jpeg):
        try:
            dims = parser(head)
        except (struct.error, IndexError, ValueError):
            continue
        if dims:
            return dims
    return None


def _png(head: bytes) -> Optional[Tuple[int, int]]:
    if not head.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def _gif(head: bytes) -> Optional[Tuple[int, int]]:
    if head[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", head[6:10])
    return int(width), int(height)


def _bmp(head: bytes) -> Optional[Tuple[int, int]]:
    if not head.startswith(b"BM"):
        return None
    width, height = struct.unpack("<ii", head[18:26])
    # A negative height means a top-down DIB; the magnitude is still the height.
    return int(abs(width)), int(abs(height))


def _webp(head: bytes) -> Optional[Tuple[int, int]]:
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        return None
    chunk = head[12:16]
    if chunk == b"VP8 ":
        # Lossy: 3-byte frame tag, 3-byte sync code, then 14-bit dims.
        width, height = struct.unpack("<HH", head[26:30])
        return int(width & 0x3FFF), int(height & 0x3FFF)
    if chunk == b"VP8L":
        bits = struct.unpack("<I", head[21:25])[0]
        return int((bits & 0x3FFF) + 1), int(((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        width = int.from_bytes(head[24:27], "little") + 1
        height = int.from_bytes(head[27:30], "little") + 1
        return width, height
    return None


# Start-of-frame markers. Excludes DHT/DAC/RST/SOS, which are not frame headers.
_JPEG_SOF = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def _jpeg(head: bytes) -> Optional[Tuple[int, int]]:
    if head[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(head)
    while i < n - 9:
        if head[i] != 0xFF:
            i += 1
            continue
        marker = head[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan - no frame header past here
            return None
        seg_len = struct.unpack(">H", head[i + 2:i + 4])[0]
        if marker in _JPEG_SOF:
            height, width = struct.unpack(">HH", head[i + 5:i + 9])
            return int(width), int(height)
        i += 2 + seg_len
    return None


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------

class ImageDimensionResolver:
    """Resolves and caches image dimensions. Cache is unbounded per process by design:
    one small tuple per distinct image, and the process is restarted between batches."""

    def __init__(self, settings: Settings, client: Optional[httpx.Client] = None):
        self._settings = settings
        self._client = client or httpx.Client(
            timeout=settings.image_probe_timeout_s, follow_redirects=True
        )
        self._cache: Dict[str, Tuple[int, int]] = {}

    def resolve(self, image_url: str, task_data: Optional[dict] = None) -> ImageDims:
        from_data = _dims_from_task_data(task_data)
        if from_data:
            return ImageDims(from_data[0], from_data[1], "task_data")

        if self._settings.image_dim_source == "probe":
            cached = self._cache.get(image_url)
            if cached:
                return ImageDims(cached[0], cached[1], "probe")
            probed = self._probe(image_url)
            if probed:
                self._cache[image_url] = probed
                return ImageDims(probed[0], probed[1], "probe")

        # IMAGE_DIM_SOURCE=fixed is an explicit operator decision to trust the
        # configured size, so it is reliable by definition. A *failed probe* is
        # not the same thing: nobody chose that number, and using it anyway is
        # how a 640x480 image gets served as 1920x1080.
        deliberate = self._settings.image_dim_source == "fixed"
        if deliberate:
            log.debug(
                "Using the configured %sx%s for %s (IMAGE_DIM_SOURCE=fixed).",
                self._settings.default_image_width, self._settings.default_image_height, image_url,
            )
        else:
            log.error(
                "Could not determine the real dimensions of %s, and IMAGE_DIM_SOURCE=probe "
                "means nobody chose the %sx%s default for it. Refusing to guess: this task "
                "will be served WITHOUT a pre-annotation rather than with one at the wrong "
                "scale. Fix the image URL, or set IMAGE_DIM_SOURCE=fixed if the default is "
                "genuinely correct for every image.",
                image_url,
                self._settings.default_image_width,
                self._settings.default_image_height,
            )

        return ImageDims(
            self._settings.default_image_width,
            self._settings.default_image_height,
            "fixed",
            deliberate,
        )

    def _local_asset(self, url_path: str) -> Optional[bytes]:
        """
        Header bytes for a URL under this service's own `/assets/` mount.

        Returns None when the URL is not one of ours. Path traversal is blocked
        by resolving and confirming the result is still inside the assets
        directory — the assets URL is attacker-influenced in the sense that it
        arrives inside a `QueueTask` from another service.
        """
        prefix = self._settings.assets_url_prefix
        if not url_path or not url_path.startswith(prefix):
            return None

        root = Path(self._settings.assets_dir).resolve()
        candidate = (root / unquote(url_path[len(prefix):])).resolve()
        if root not in candidate.parents and candidate != root:
            log.warning("Refusing an assets path that escapes %s: %s", root, url_path)
            return None
        if not candidate.is_file():
            log.warning("No such local asset: %s", candidate)
            return None

        with candidate.open("rb") as fh:
            return fh.read(_HEADER_BYTES)

    def _probe(self, image_url: str) -> Optional[Tuple[int, int]]:
        head = self._read_head(image_url)
        if not head:
            return None
        dims = parse_dimensions(head)
        if not dims:
            log.warning("Could not read image dimensions from the header of %s", image_url)
        return dims

    def _read_head(self, image_url: str) -> Optional[bytes]:
        parsed = urlparse(image_url)

        # An image this service serves itself: read it off disk instead of
        # fetching our own listening socket. Besides being a pointless round
        # trip, the HTTP path fails under a test client (nothing is listening)
        # and needs the container to be able to resolve its own published URL.
        local = self._local_asset(parsed.path)
        if local is not None:
            return local

        # A single-letter "scheme" is a Windows drive letter, not a URL scheme:
        # urlparse("C:\\images\\frame.png") reports scheme="c". Without this,
        # every local image on Windows is rejected as an unsupported scheme and
        # silently falls back to the default dimensions - which is the standard
        # local-dev setup (STORAGE_PROVIDER=local).
        if parsed.scheme in ("", "file") or len(parsed.scheme) == 1:
            return _read_local_head(parsed, image_url)

        if parsed.scheme not in ("http", "https"):
            log.warning("Cannot probe %s: unsupported scheme %r", image_url, parsed.scheme)
            return None

        try:
            # Ask for a range; servers that ignore it just send more, which is fine.
            response = self._client.get(
                image_url, headers={"Range": f"bytes=0-{_HEADER_BYTES - 1}"}
            )
            if response.status_code >= 400:
                log.warning("Probe of %s returned HTTP %s", image_url, response.status_code)
                return None
            return response.content[:_HEADER_BYTES]
        except httpx.RequestError as exc:
            log.warning("Probe of %s failed: %s", image_url, exc)
            return None


def _read_local_head(parsed, image_url: str) -> Optional[bytes]:
    # Only a real file:// URL has its path in `parsed.path`; a bare filesystem
    # path must be used verbatim, or a Windows drive letter is lost.
    raw = parsed.path if parsed.scheme == "file" else image_url
    path = Path(unquote(raw))
    # file:///C:/x on Windows parses to /C:/x
    if path.drive == "" and len(path.parts) > 1 and path.parts[1].endswith(":"):
        path = Path(*path.parts[1:])
    if not path.exists():
        log.warning("Cannot probe %s: no such file", path)
        return None
    with path.open("rb") as fh:
        return fh.read(_HEADER_BYTES)


def _dims_from_task_data(task_data: Optional[dict]) -> Optional[Tuple[int, int]]:
    if not task_data:
        return None
    for w_key, h_key in (
        ("width", "height"),
        ("image_width", "image_height"),
        ("original_width", "original_height"),
    ):
        width, height = task_data.get(w_key), task_data.get(h_key)
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            if width > 0 and height > 0:
                return int(width), int(height)
    return None
