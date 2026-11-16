"""Turn user inputs (paths, URLs, bytes, PIL images, text) into chat content parts."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Sequence, Union

IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

ImageLike = Union[str, Path, bytes, Any]  # Any = PIL.Image.Image


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    raise ValueError("Could not detect image type from bytes (supported: png, jpeg, webp, gif, bmp).")


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_url(image: ImageLike) -> str:
    """Return a URL the server accepts: http(s) and data URLs pass through,
    local files / bytes / PIL images are inlined as base64."""
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image

    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        mime = IMAGE_SUFFIXES.get(path.suffix.lower())
        data = path.read_bytes()
        return _data_url(data, mime or _sniff_mime(data))

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        return _data_url(data, _sniff_mime(data))

    if hasattr(image, "save") and hasattr(image, "mode"):  # PIL.Image.Image
        buf = io.BytesIO()
        img = image if image.mode in ("RGB", "RGBA", "L") else image.convert("RGB")
        img.save(buf, format="PNG")
        return _data_url(buf.getvalue(), "image/png")

    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def build_content(
    prompt: str,
    image: ImageLike | Sequence[ImageLike] | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Images first, then the item text (if any), then the task prompt."""
    parts: list[dict[str, Any]] = []
    if image is not None:
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        for img in images:
            parts.append({"type": "image_url", "image_url": {"url": image_url(img)}})
    if text is not None:
        parts.append({"type": "text", "text": f"<input>\n{text}\n</input>"})
    parts.append({"type": "text", "text": prompt})
    return parts


def is_image_path(p: str | Path) -> bool:
    return Path(p).suffix.lower() in IMAGE_SUFFIXES
