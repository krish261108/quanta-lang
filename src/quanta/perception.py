"""Turning files and the computer environment into model-readable content.

* text, code, markup, data   -> text (truncated with an explicit marker)
* images                      -> image content blocks (the model sees them)
* PDFs                        -> document content blocks
* video                       -> evenly spaced key frames (via ffmpeg) + metadata
* audio                       -> transcript if a transcriber is supplied,
                                 otherwise metadata and an explicit note that
                                 the content was NOT perceived
* anything else               -> size, hash, magic bytes

Perception never silently pretends: when a modality cannot be perceived, the
result says so, so downstream claims cannot be grounded on it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

TEXT_EXT = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash", ".zsh", ".sql",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".xml", ".html", ".htm",
    ".css", ".tex", ".r", ".jl", ".m", ".lua", ".pl", ".ipynb", ".log", ".env.example", ".dockerfile",
}
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 32 * 1024 * 1024


@dataclass
class Percept:
    kind: str
    path: str
    blocks: list[dict] = field(default_factory=list)   # content blocks for a model
    summary: str = ""
    perceived: bool = True
    meta: dict = field(default_factory=dict)


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_textual(path: Path) -> bool:
    with path.open("rb") as f:
        head = f.read(4096)
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def ffprobe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return {"error": "ffprobe not available"}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout or "{}")
        fmt = data.get("format", {})
        return {"duration_s": float(fmt.get("duration", 0) or 0), "format": fmt.get("format_name"),
                "streams": [{"type": s.get("codec_type"), "codec": s.get("codec_name"),
                             "width": s.get("width"), "height": s.get("height")}
                            for s in data.get("streams", [])]}
    except (subprocess.SubprocessError, ValueError) as e:
        return {"error": str(e)}


def extract_video_frames(path: Path, n_frames: int = 4) -> list[bytes]:
    """Evenly spaced JPEG key frames via ffmpeg (empty list if unavailable)."""
    if not shutil.which("ffmpeg"):
        return []
    duration = ffprobe(path).get("duration_s") or 0.0
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_frames):
            t = duration * (i + 0.5) / n_frames if duration else 0.0
            out = Path(tmp) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                            "-frames:v", "1", "-vf", "scale='min(1024,iw)':-2", str(out)],
                           capture_output=True, timeout=60)
            if out.exists() and out.stat().st_size:
                frames.append(out.read_bytes())
    return frames


def perceive(path: str | Path, *, max_chars: int = 60_000, video_frames: int = 4,
             transcriber: Callable[[Path], str] | None = None) -> Percept:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    size = p.stat().st_size
    meta = {"bytes": size, "sha256": _sha256(p)}

    if ext in IMAGE_TYPES:
        if size > MAX_IMAGE_BYTES:
            return Percept("image", str(p), summary=f"image too large to attach ({size} bytes)",
                           perceived=False, meta=meta)
        block = {"type": "image", "source": {"type": "base64", "media_type": IMAGE_TYPES[ext],
                                             "data": _b64(p.read_bytes())}}
        return Percept("image", str(p), [block], f"image {p.name} ({size} bytes)", meta=meta)

    if ext == ".pdf":
        if size > MAX_PDF_BYTES:
            return Percept("pdf", str(p), summary="PDF too large to attach", perceived=False, meta=meta)
        block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                "data": _b64(p.read_bytes())}}
        return Percept("pdf", str(p), [block], f"PDF {p.name} ({size} bytes)", meta=meta)

    if ext in VIDEO_EXT:
        meta.update(ffprobe(p))
        frames = extract_video_frames(p, video_frames)
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                               "data": _b64(f)}} for f in frames]
        note = (f"video {p.name}: {len(frames)} key frames attached; metadata {json.dumps(meta)}. "
                "Audio track NOT perceived." if frames else
                f"video {p.name}: frames could not be extracted (ffmpeg missing or failed).")
        blocks.insert(0, {"type": "text", "text": note})
        return Percept("video", str(p), blocks, note, perceived=bool(frames), meta=meta)

    if ext in AUDIO_EXT:
        meta.update(ffprobe(p))
        if transcriber is not None:
            text = transcriber(p)
            return Percept("audio", str(p), [{"type": "text", "text": f"Transcript of {p.name}:\n{text}"}],
                           f"audio {p.name} transcribed", meta=meta)
        note = (f"audio {p.name}: metadata {json.dumps(meta)}. Content NOT perceived "
                "(no transcriber configured); do not make claims about what it says.")
        return Percept("audio", str(p), [{"type": "text", "text": note}], note, perceived=False, meta=meta)

    if ext in TEXT_EXT or _looks_textual(p):
        text = p.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + f"\n\n[... truncated: showed {max_chars} of {len(text)} characters]"
        return Percept("text", str(p), [{"type": "text", "text": text}],
                       f"text {p.name} ({size} bytes{', truncated' if truncated else ''})", meta=meta)

    with p.open("rb") as f:
        meta["magic"] = f.read(16).hex()
    meta["mime_guess"] = mimetypes.guess_type(p.name)[0]
    note = f"binary file {p.name}: {json.dumps(meta)}. Content NOT perceived."
    return Percept("binary", str(p), [{"type": "text", "text": note}], note, perceived=False, meta=meta)


def observe_environment(workspace: str | Path, *, max_entries: int = 200) -> dict:
    """A bounded snapshot of the computer environment the agent operates in."""
    ws = Path(workspace)
    entries = []
    for root, dirs, files in os.walk(ws):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__"))
        rel = Path(root).relative_to(ws)
        for name in sorted(files):
            entries.append(str(rel / name))
            if len(entries) >= max_entries:
                break
        if len(entries) >= max_entries:
            break
    git = None
    if (ws / ".git").exists() and shutil.which("git"):
        r = subprocess.run(["git", "status", "--short", "--branch"], cwd=ws, capture_output=True,
                           text=True, timeout=20)
        git = r.stdout[:4000]
    tools = sorted(t for t in ("git", "python3", "pip", "node", "npm", "cargo", "go", "java", "ffmpeg",
                               "docker", "make", "gcc", "sqlite3", "curl") if shutil.which(t))
    return {"platform": platform.platform(), "python": sys.version.split()[0], "cwd": str(ws),
            "cpu_count": os.cpu_count(), "files": entries, "files_truncated": len(entries) >= max_entries,
            "git_status": git, "available_cli_tools": tools}
