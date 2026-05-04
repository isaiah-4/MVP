from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOSTS = {
    "youtu.be",
    "www.youtube.com",
    "youtube.com",
    "m.youtube.com",
}
YOUTUBE_DOWNLOAD_FORMAT = "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_VIDEOS_ROOT = (PROJECT_ROOT / "Input_vids").resolve()
STUB_FILE_SUFFIX = ".msgpack.gz"


@dataclass(frozen=True)
class VideoRunPaths:
    source_key: str
    base_source_key: str
    input_path: Path
    output_path: Path
    player_stub_path: Path
    ball_stub_path: Path
    court_stub_path: Path


class InputSourceValidationError(ValueError):
    """Raised when a requested input source is not allowed."""


def prepare_video_source(source: str, run_suffix: str | None = None) -> VideoRunPaths:
    validated_source = validate_input_source(source)

    if is_youtube_url(validated_source):
        base_source_key = build_source_key(validated_source)
        download_path = PROJECT_ROOT / "Input_vids" / "downloads" / f"{base_source_key}.mp4"
        input_path = ensure_youtube_download(validated_source, download_path)
    else:
        input_path = resolve_local_input_path(validated_source)
        base_source_key = build_source_key(validated_source, resolved_path=input_path)

    source_key = base_source_key
    if run_suffix:
        source_key = f"{base_source_key}_{sanitize_name(run_suffix)}"

    return VideoRunPaths(
        source_key=source_key,
        base_source_key=base_source_key,
        input_path=input_path,
        output_path=PROJECT_ROOT / "Output_vids" / f"{source_key}_output.mp4",
        player_stub_path=PROJECT_ROOT / "stubs" / f"{source_key}_player_tracker{STUB_FILE_SUFFIX}",
        ball_stub_path=PROJECT_ROOT / "stubs" / f"{source_key}_ball_tracker{STUB_FILE_SUFFIX}",
        court_stub_path=PROJECT_ROOT / "stubs" / f"{source_key}_court_keypoints{STUB_FILE_SUFFIX}",
    )


def validate_input_source(source: str, *, require_exists: bool = True) -> str:
    normalized_source = str(source or "").strip()
    if not normalized_source:
        raise InputSourceValidationError("Input source is required.")

    if is_youtube_url(normalized_source):
        if extract_youtube_id(normalized_source) is None:
            raise InputSourceValidationError(
                "Input source must be a file inside Input_vids or a supported YouTube URL."
            )
        if shutil.which("yt-dlp") is None and importlib.util.find_spec("yt_dlp") is None:
            raise InputSourceValidationError(
                "YouTube analysis is unavailable on this server because yt-dlp is not installed."
            )
        return normalized_source

    input_path = resolve_local_input_path(normalized_source)
    if not _is_relative_to(input_path, INPUT_VIDEOS_ROOT):
        raise InputSourceValidationError(
            "Input source must be a file inside Input_vids or a supported YouTube URL."
        )
    if require_exists and (not input_path.exists() or not input_path.is_file()):
        raise InputSourceValidationError("Input video was not found.")

    return str(input_path.relative_to(PROJECT_ROOT))


def resolve_local_input_path(source: str) -> Path:
    input_path = Path(str(source or "").strip()).expanduser()
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    else:
        input_path = input_path.resolve()
    return input_path


def build_source_key(source: str, resolved_path: Path | None = None) -> str:
    if is_youtube_url(source):
        video_id = extract_youtube_id(source)
        if video_id is None:
            raise ValueError(f"Could not extract a YouTube video id from: {source}")
        return video_id

    source_path = resolved_path if resolved_path is not None else Path(source).expanduser()
    resolved_source_path = source_path.resolve()
    base_name = sanitize_name(resolved_source_path.stem or resolved_source_path.name or "video")
    path_digest = hashlib.blake2b(
        str(resolved_source_path).encode("utf-8"),
        digest_size=6,
    ).hexdigest()
    return f"{base_name}_{path_digest}"


def ensure_youtube_download(url: str, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path is not None:
        download_with_cli(yt_dlp_path, url, output_path)
        return output_path

    if importlib.util.find_spec("yt_dlp") is not None:
        download_with_module(url, output_path)
        return output_path

    raise RuntimeError(
        "yt-dlp is required to download YouTube videos. Install it and rerun the command."
    )


def download_with_cli(yt_dlp_path: str, url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_template = output_path.with_suffix(".%(ext)s")

    command = [
        yt_dlp_path,
        "--format",
        YOUTUBE_DOWNLOAD_FORMAT,
        "--merge-output-format",
        "mp4",
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "--output",
        str(output_template),
        "--print",
        "after_move:filepath",
        url,
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown yt-dlp error."
        raise RuntimeError(f"Failed to download YouTube video: {error_text}")

    downloaded_path = parse_download_path(result.stdout, output_path)
    finalize_downloaded_path(downloaded_path, output_path)


def download_with_module(url: str, output_path: Path) -> None:
    import yt_dlp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_template = output_path.with_suffix(".%(ext)s")

    options = {
        "format": YOUTUBE_DOWNLOAD_FORMAT,
        "merge_output_format": "mp4",
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(output_template),
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        downloaded_path = Path(downloader.prepare_filename(info)).expanduser()

    if downloaded_path.suffix != ".mp4":
        mp4_candidate = downloaded_path.with_suffix(".mp4")
        if mp4_candidate.exists():
            downloaded_path = mp4_candidate

    finalize_downloaded_path(downloaded_path, output_path)


def finalize_downloaded_path(downloaded_path: Path | None, output_path: Path) -> None:
    if downloaded_path is None or not downloaded_path.exists():
        raise RuntimeError("YouTube download completed but the output file was not found.")

    if downloaded_path != output_path:
        output_path.unlink(missing_ok=True)
        shutil.move(str(downloaded_path), str(output_path))


def parse_download_path(stdout: str, fallback_path: Path) -> Path | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if lines:
        return Path(lines[-1]).expanduser()

    if fallback_path.exists():
        return fallback_path

    matches = sorted(fallback_path.parent.glob(f"{fallback_path.stem}.*"))
    if matches:
        return matches[-1]

    return None


def is_youtube_url(source: str) -> bool:
    parsed_url = urlparse(source)
    return parsed_url.scheme in {"http", "https"} and parsed_url.netloc.lower() in YOUTUBE_HOSTS


def extract_youtube_id(source: str) -> str | None:
    parsed_url = urlparse(source)
    host = parsed_url.netloc.lower()

    if host == "youtu.be":
        candidate = parsed_url.path.strip("/").split("/")[0]
    elif parsed_url.path == "/watch":
        candidate = parse_qs(parsed_url.query).get("v", [None])[0]
    else:
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
            candidate = path_parts[1]
        else:
            candidate = None

    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate

    return None


def sanitize_name(value: str) -> str:
    cleaned_value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned_value or "video"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
