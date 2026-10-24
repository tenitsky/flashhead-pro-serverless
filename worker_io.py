"""Inputs and outputs of the FlashHead worker.

In: the portrait and voiceover a job names - an https URL, base64 (plain or a data URI),
a file on the endpoint's network volume (/runpod-volume/...), or "sample" for the files
bundled with the image.

Out: the finished mp4 - uploaded to an S3-compatible bucket (BUCKET_ENDPOINT_URL set),
saved on the network volume (one is attached), or returned as base64 in the job result
(small videos only: RunPod job results are not meant for large files).

Nothing here imports torch, so it is tested on any machine (tests/).
"""
import base64
import binascii
import os
import re
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = {"portrait": os.path.join(HERE, "samples", "sample_portrait.png"),
           "audio": os.path.join(HERE, "samples", "sample_voiceover.mp3")}
VOLUME = "/runpod-volume"
OUTPUT_MODES = ("auto", "base64", "bucket", "volume")


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class InputError(ValueError):
    """A problem with the job's input; reported to the caller as the job's error."""


# ---------------------------------------------------------------------------- inputs
def fetch(value, kind, dest_dir):
    """The portrait or voiceover a job names -> a local file path."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"'{kind}' is required: an https URL, base64 data, a "
                         f"{VOLUME}/... path or \"sample\".")
    value = value.strip()
    if value == "sample":
        return SAMPLES[kind]
    limit = env_float("MAX_INPUT_MB", 200) * 2**20
    if value.startswith(VOLUME + "/"):
        root, path = os.path.realpath(VOLUME), os.path.realpath(value)
        if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
            raise InputError(f"'{kind}': {value} is not a file on the network volume.")
        return path
    path = os.path.join(dest_dir, kind)
    if re.match(r"^https?://", value, re.I):
        download(value, path, kind, limit)
        return path
    if value.startswith("data:"):
        value = value.partition(",")[2]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise InputError(f"'{kind}' is not an https URL, base64 data, a {VOLUME}/... path "
                         f"or \"sample\".") from None
    if not data:
        raise InputError(f"'{kind}' is empty.")
    if len(data) > limit:
        raise InputError(f"'{kind}' is {len(data) / 2**20:.0f} MB; the limit is "
                         f"{limit / 2**20:.0f} MB (MAX_INPUT_MB).")
    with open(path, "wb") as f:
        f.write(data)
    return path


def download(url, path, kind, limit):
    import requests
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            size = 0
            with open(path, "wb") as f:
                for block in r.iter_content(1 << 20):
                    size += len(block)
                    if size > limit:
                        raise InputError(f"'{kind}' at {url} is over the "
                                         f"{limit / 2**20:.0f} MB limit (MAX_INPUT_MB).")
                    f.write(block)
    except requests.RequestException as e:
        raise InputError(f"Could not download '{kind}' from {url}: {e}") from None
    if size == 0:
        raise InputError(f"'{kind}' at {url} is empty.")


def load_image(path):
    """Any image file -> RGB uint8 array."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError) as e:
        raise InputError(f"The portrait is not an image this worker can read: {e}") from None


def load_audio(path, sample_rate=48000):
    """Any audio (or video) file ffmpeg reads -> ([2, samples] float32, sample_rate)."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2",
                        "-ar", str(sample_rate), "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise InputError("The voiceover is not audio this worker can read: "
                         + p.stderr.decode(errors="replace").strip()[-300:])
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if data.size < 2 * sample_rate // 2:
        raise InputError("The voiceover is shorter than half a second.")
    return data.reshape(-1, 2).T.copy(), sample_rate


# ---------------------------------------------------------------------------- outputs
def bucket_configured():
    return bool(os.environ.get("BUCKET_ENDPOINT_URL"))


def volume_attached():
    return os.path.isdir(VOLUME)


def resolve_mode(mode):
    """"auto" -> the first of bucket, volume, base64 that this endpoint can do."""
    mode = (mode or "auto").strip().lower()
    if mode not in OUTPUT_MODES:
        raise InputError(f"'output' must be one of {', '.join(OUTPUT_MODES)}.")
    if mode == "auto":
        return "bucket" if bucket_configured() else "volume" if volume_attached() else "base64"
    if mode == "bucket" and not bucket_configured():
        raise InputError("output 'bucket' needs BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID "
                         "and BUCKET_SECRET_ACCESS_KEY set on the endpoint.")
    if mode == "volume" and not volume_attached():
        raise InputError("output 'volume' needs a network volume attached to the endpoint.")
    return mode


def check_base64_fits(mode, audio_seconds):
    """Refuse before rendering, not after, when a base64 result would be too long."""
    # FlashHead's mp4 runs 6-22 MB per minute (crf 17), and base64 adds a third.
    limit = env_float("MAX_BASE64_SECONDS", 30)
    if mode == "base64" and audio_seconds > limit:
        raise InputError(
            f"The voiceover is {audio_seconds:.0f} s. Videos longer than {limit:g} s are not "
            f"returned as base64 (MAX_BASE64_SECONDS): attach a network volume or set "
            f"BUCKET_ENDPOINT_URL on the endpoint, or send a shorter voiceover.")


def deliver(video, job_id, mode):
    """The finished mp4 -> the part of the job result that hands it over."""
    size_mb = round(os.path.getsize(video) / 2**20, 2)
    name = f"{job_id}.mp4"
    if mode == "bucket":
        from runpod.serverless.utils import rp_upload
        url = rp_upload.upload_file_to_bucket(file_name=name, file_location=video)
        return {"video_url": url, "video_mb": size_mb}
    if mode == "volume":
        folder = os.path.join(VOLUME, "flashhead", "outputs")
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, name)
        shutil.copyfile(video, dest)
        return {"video_path": dest, "video_mb": size_mb}
    with open(video, "rb") as f:
        return {"video_base64": base64.b64encode(f.read()).decode("ascii"),
                "video_mime": "video/mp4", "video_mb": size_mb}
