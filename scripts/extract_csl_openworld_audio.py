#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://huggingface.co/datasets/hulala/CSL-OpenWorld/resolve/main"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio for CSL-OpenWorld clips and store it with the same "
            "basename as the source video."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/annotations"),
        help="Directory containing annotation_train.txt / annotation_dev.txt / annotation_test.txt.",
    )
    parser.add_argument(
        "--video-index",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/video_index.tsv"),
        help="TSV file mapping video_key to repo-relative video path.",
    )
    parser.add_argument(
        "--video-source",
        default="clips",
        choices=["clips", "source_url"],
        help="Read audio from dataset clip files or from the original source_url pages.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help=(
            "Local root for existing videos. When set, the script looks for "
            "<videos_root>/<video_rel_path> before trying remote download."
        ),
    )
    parser.add_argument(
        "--source-video-cache-dir",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/source_videos"),
        help="Directory used to cache original source videos when --video-source source_url is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/audio"),
        help="Directory where extracted audio files will be stored.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/videos_cache"),
        help="Directory used to cache downloaded videos when --download-missing is enabled.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Datasets/CSL-OpenWorld/audio_manifest.csv"),
        help="CSV manifest for extraction results.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing videos from Hugging Face using repo-relative video paths.",
    )
    parser.add_argument(
        "--source-base-url",
        default=DEFAULT_BASE_URL,
        help="Base resolve URL for CSL-OpenWorld files on Hugging Face.",
    )
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Output audio format.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output audio sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Output audio channel count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N clips after loading annotations.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose target audio file already exists.",
    )
    parser.add_argument(
        "--source-domain-allowlist",
        nargs="+",
        default=None,
        help="Only process source_url entries whose domain is in this allowlist.",
    )
    parser.add_argument(
        "--yt-dlp-bin",
        default="yt-dlp",
        help="yt-dlp executable name when using --video-source source_url.",
    )
    parser.add_argument(
        "--yt-dlp-pythonpath",
        default=None,
        help="Optional PYTHONPATH that makes `python -m yt_dlp` available.",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Optional cookies.txt passed to yt-dlp for source page access.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Optional browser name passed to yt-dlp via --cookies-from-browser.",
    )
    return parser.parse_args()


def annotation_path(annotation_dir: Path, split: str) -> Path:
    return annotation_dir / f"annotation_{split}.txt"


def normalize_timestamp(ts: str) -> str:
    return ts.replace(":", ".")


def parse_annotation_line(line: str, split: str):
    line = line.strip()
    if not line:
        return None

    parts = line.split(None, 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected annotation format: {line}")

    sample_id, start_time, end_time, tail = parts
    tail_parts = tail.rsplit(None, 1)
    if len(tail_parts) == 2:
        text, source_url = tail_parts
    else:
        text, source_url = tail, ""
    video_key = f"{sample_id}_{normalize_timestamp(start_time)}_{normalize_timestamp(end_time)}"

    return {
        "split": split,
        "sample_id": sample_id,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_url": source_url,
        "video_key": video_key,
        "video_filename": f"{video_key}.mp4",
    }


def load_annotations(annotation_dir: Path, splits, limit=None):
    entries = []
    allowlist = None
    args = None
    if hasattr(load_annotations, "_args"):
        args = load_annotations._args
        if args.source_domain_allowlist:
            allowlist = set(args.source_domain_allowlist)
    for split in splits:
        ann_path = annotation_path(annotation_dir, split)
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        with ann_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                entry = parse_annotation_line(raw_line, split)
                if entry is None:
                    continue
                if allowlist is not None:
                    domain = urlparse(entry["source_url"]).netloc
                    if domain not in allowlist:
                        continue
                entries.append(entry)
                if limit is not None and len(entries) >= limit:
                    return entries
    return entries


def load_video_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(f"Missing video index: {index_path}")

    index = {}
    with index_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            video_key, rel_path = row
            index[video_key] = rel_path
    return index


def download_file(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        with target_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def safe_source_name(url: str) -> str:
    parsed = urlparse(url)
    raw = f"{parsed.netloc}_{parsed.path}".strip("_")
    raw = raw.replace("/", "_")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    raw = raw.strip("._")
    if not raw:
        raw = "source"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{raw[:80]}_{digest}"


def build_yt_dlp_command(args, output_template: str, url: str):
    if args.yt_dlp_pythonpath:
        command = [sys.executable, "-m", "yt_dlp"]
    else:
        command = [args.yt_dlp_bin]

    command.extend(
        [
            "--no-progress",
            "--no-warnings",
            "--restrict-filenames",
            "--merge-output-format",
            "mp4",
            "--print",
            "after_move:filepath",
            "-o",
            output_template,
        ]
    )

    if args.cookies_file is not None:
        command.extend(["--cookies", str(args.cookies_file)])
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])

    command.append(url)
    return command


def classify_source_error(message: str, source_url: str) -> str:
    lower = message.lower()
    domain = urlparse(source_url).netloc
    if "unsupported url" in lower:
        return "source_unsupported"
    if "captcha" in lower or "environment abnormal" in lower or "环境异常" in lower:
        return "source_captcha"
    if "http error 412" in lower or "precondition failed" in lower:
        return "source_http_412"
    if domain == "mp.weixin.qq.com":
        return "source_weixin_blocked"
    return "source_download_failed"


def download_source_video(source_url: str, args):
    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.netloc:
        return None, "no_source_url", ""

    cache_name = safe_source_name(source_url)
    cache_dir = args.source_video_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob(f"{cache_name}.*"))
    if existing:
        return existing[0], "source_cached", ""

    output_template = str(cache_dir / f"{cache_name}.%(ext)s")
    command = build_yt_dlp_command(args, output_template, source_url)
    env = os.environ.copy()
    if args.yt_dlp_pythonpath:
        env["PYTHONPATH"] = args.yt_dlp_pythonpath

    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        return None, "yt_dlp_not_found", str(exc)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)

    if result.returncode != 0:
        return None, classify_source_error(combined, source_url), combined.strip()

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for candidate in reversed(output_lines):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path, "source_downloaded", ""

    created = sorted(cache_dir.glob(f"{cache_name}.*"))
    if created:
        return created[0], "source_downloaded", ""

    return None, "source_download_missing_output", combined.strip()


def ensure_video(entry, video_index, args):
    rel_path = video_index.get(entry["video_key"])
    if rel_path is None:
        return None, "", "missing_video_index"

    local_path = None
    if args.videos_root is not None:
        candidate = args.videos_root / rel_path
        if candidate.exists():
            local_path = candidate

    if local_path is not None:
        return local_path, rel_path, "local_video"

    if not args.download_missing:
        return None, rel_path, "missing_local_video"

    cached_path = args.cache_dir / rel_path
    if not cached_path.exists():
        url = f"{args.source_base_url}/{quote(rel_path, safe='/')}"
        download_file(url, cached_path)
    return cached_path, rel_path, "downloaded_video"


def has_audio_stream(video_path: Path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def extract_audio(video_path: Path, audio_path: Path, args):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(args.channels),
        "-ar",
        str(args.sample_rate),
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()


def extract_audio_segment(video_path: Path, start_time: str, end_time: str, audio_path: Path, args):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(args.channels),
        "-ar",
        str(args.sample_rate),
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()


def main():
    args = parse_args()
    load_annotations._args = args

    try:
        entries = load_annotations(args.annotation_dir, args.splits, args.limit)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    video_index = {}
    if args.video_source == "clips":
        try:
            video_index = load_video_index(args.video_index)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.download_missing:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.video_source == "source_url":
        args.source_video_cache_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_source",
        "split",
        "sample_id",
        "start_time",
        "end_time",
        "video_key",
        "video_rel_path",
        "video_path",
        "source_cache_path",
        "audio_path",
        "status",
        "detail",
        "text",
        "source_url",
    ]

    processed = 0
    source_cache = {}
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for entry in entries:
            audio_path = args.output_dir / f"{entry['video_key']}.{args.audio_format}"
            row = {
                "video_source": args.video_source,
                "split": entry["split"],
                "sample_id": entry["sample_id"],
                "start_time": entry["start_time"],
                "end_time": entry["end_time"],
                "video_key": entry["video_key"],
                "video_rel_path": "",
                "video_path": "",
                "source_cache_path": "",
                "audio_path": str(audio_path),
                "status": "",
                "detail": "",
                "text": entry["text"],
                "source_url": entry["source_url"],
            }

            if args.skip_existing and audio_path.exists():
                row["status"] = "audio_exists"
                writer.writerow(row)
                handle.flush()
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(entries)} clips")
                continue

            if args.video_source == "clips":
                try:
                    video_path, rel_path, source_status = ensure_video(entry, video_index, args)
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    row["video_rel_path"] = video_index.get(entry["video_key"], "")
                    row["status"] = f"download_failed:{exc.__class__.__name__}"
                    row["detail"] = str(exc)
                    writer.writerow(row)
                    handle.flush()
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(entries)} clips")
                    continue

                row["video_rel_path"] = rel_path
                row["video_path"] = "" if video_path is None else str(video_path)

                if video_path is None:
                    row["status"] = source_status
                    writer.writerow(row)
                    handle.flush()
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(entries)} clips")
                    continue

                if not has_audio_stream(video_path):
                    row["status"] = "no_audio_stream"
                    writer.writerow(row)
                    handle.flush()
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(entries)} clips")
                    continue

                ok, detail = extract_audio(video_path, audio_path, args)
                row["status"] = "ok" if ok else "ffmpeg_failed"
                row["detail"] = "" if ok else detail
            else:
                source_url = entry["source_url"]
                if source_url in source_cache:
                    source_video_path, source_status, source_error = source_cache[source_url]
                else:
                    source_video_path, source_status, source_error = download_source_video(source_url, args)
                    source_cache[source_url] = (source_video_path, source_status, source_error)

                row["status"] = source_status
                row["detail"] = source_error
                row["source_cache_path"] = "" if source_video_path is None else str(source_video_path)

                if source_video_path is None:
                    writer.writerow(row)
                    handle.flush()
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(entries)} clips")
                    continue

                if not has_audio_stream(source_video_path):
                    row["status"] = "source_video_no_audio_stream"
                    writer.writerow(row)
                    handle.flush()
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(entries)} clips")
                    continue

                ok, detail = extract_audio_segment(
                    source_video_path,
                    entry["start_time"],
                    entry["end_time"],
                    audio_path,
                    args,
                )
                row["video_path"] = str(source_video_path)
                row["status"] = "ok" if ok else "ffmpeg_failed"
                row["detail"] = "" if ok else detail

            writer.writerow(row)
            handle.flush()
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(entries)} clips")

    print(f"Processed {processed} clips")
    print(f"Manifest saved to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
