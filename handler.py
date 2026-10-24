"""
RunPod serverless worker: SoulX-FlashHead Pro talking heads.

One portrait + a voiceover of any length -> a 512 x 512, 25 fps talking-head mp4 with the
original audio. The model loads once per worker, before the first job.

Job input (all but portrait and audio optional):
    portrait           https URL | base64 / data URI | /runpod-volume/... path | "sample"
    audio              same; any format ffmpeg reads (mp3, wav, m4a, even a video)
    anchor             "continuous" (default) | "refresh_at_pauses" | "reset_every_n"
                       | "reset_every_chunk"
    reset_seconds      40   spacing for refresh_at_pauses / reset_every_n
    framing            "face" (default: a square 2x the face width, as FlashHead was
                       trained) | "whole" (center crop of the whole image)
    face_ratio         2.0  square size in face widths ("face" framing)
    seed               42
    color_correction   1.0  0-1, pulls every chunk's colours toward the portrait
    max_seconds        0    render only the first N seconds of the voiceover (0 = all)
    output             "auto" (default) | "bucket" | "volume" | "base64"

Result: video_url (bucket) | video_path (volume) | video_base64, plus video_mb, the
timing (render_seconds, seconds_per_video_second, ...), gpu and peak_vram_gb.

Locally (the model on this machine's GPU):
    python handler.py --test_input '{"input": {"portrait": "sample", "audio": "sample",
                                               "max_seconds": 5}}'
"""
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))

import worker_io  # noqa: E402
from worker_io import InputError  # noqa: E402

MODELS = os.environ.get("FLASHHEAD_MODELS", "/models")
ANCHORS = {"continuous": "ANCHOR_CONTINUOUS", "refresh_at_pauses": "ANCHOR_PAUSES",
           "reset_every_n": "ANCHOR_EVERY_N", "reset_every_chunk": "ANCHOR_EVERY_CHUNK"}
FRAMINGS = {"face": "FRAMING_FACE", "whole": "FRAMING_WHOLE"}
PROGRESS_EVERY = 5.0          # seconds between progress updates to the job status


def env_flag(name, default=False):
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


_model = None


def model():
    """(pipeline, motion_frames) - loaded once per worker and reused by every job."""
    global _model
    if _model is None:
        import engine
        t = time.time()
        _model = engine.load_pipeline(os.path.join(MODELS, "SoulX-FlashHead-1_3B"),
                                      os.path.join(MODELS, "wav2vec2-base-960h"),
                                      model_type="pro", compile_model=env_flag("FLASHHEAD_COMPILE"))
        print(f"[worker] model ready in {time.time() - t:.1f} s", flush=True)
    return _model


def number(inp, key, default, lo, hi, cast=float):
    value = inp.get(key, default)
    if value is None:
        value = default
    try:
        value = cast(value)
    except (TypeError, ValueError):
        raise InputError(f"'{key}' must be a number.") from None
    if not lo <= value <= hi:
        raise InputError(f"'{key}' must be between {lo} and {hi}.")
    return value


def choice(inp, key, options, default):
    value = str(inp.get(key) or default).strip().lower()
    if value not in options:
        raise InputError(f"'{key}' must be one of {', '.join(options)}.")
    return value


def settings(inp):
    """The job's options, checked before anything is downloaded or rendered."""
    if not isinstance(inp, dict):
        raise InputError("The job input must be a JSON object.")
    return {"anchor": choice(inp, "anchor", ANCHORS, "continuous"),
            "framing": choice(inp, "framing", FRAMINGS, "face"),
            "reset_seconds": number(inp, "reset_seconds", 40.0, 5.0, 600.0),
            "face_ratio": number(inp, "face_ratio", 2.0, 1.2, 4.0),
            "seed": number(inp, "seed", 42, 0, 2**31 - 1, int),
            "color_correction": number(inp, "color_correction", 1.0, 0.0, 1.0),
            "max_seconds": number(inp, "max_seconds", 0.0, 0.0, 24 * 3600.0),
            "output": worker_io.resolve_mode(inp.get("output"))}


class Progress:
    """Engine chunk callback -> RunPod job progress, at most every few seconds."""

    def __init__(self, job):
        self.job, self.last = job, 0.0

    def __call__(self, i, total, frame=None):
        now = time.time()
        if now - self.last < PROGRESS_EVERY and i + 1 < total:
            return
        self.last = now
        try:
            import runpod
            runpod.serverless.progress_update(
                self.job, {"chunk": i + 1, "chunks": total, "percent": round(100 * (i + 1) / total, 1)})
        except Exception as e:      # progress is a nicety; never fail a render over it
            print(f"[worker] progress update failed: {e}", flush=True)


def handler(job):
    inp = job.get("input") or {}
    job_id = str(job.get("id") or "local")
    work = tempfile.mkdtemp(prefix="flashhead-")
    try:
        s = settings(inp)
        portrait_file = worker_io.fetch(inp.get("portrait"), "portrait", work)
        audio_file = worker_io.fetch(inp.get("audio"), "audio", work)
        rgb = worker_io.load_image(portrait_file)
        waveform, rate = worker_io.load_audio(audio_file)
        if s["max_seconds"]:
            waveform = waveform[:, :int(s["max_seconds"] * rate)]
        seconds = waveform.shape[1] / rate
        limit = worker_io.env_float("MAX_AUDIO_SECONDS", 3 * 3600)
        if seconds > limit:
            raise InputError(f"The voiceover is {seconds / 60:.0f} min; this endpoint renders up to "
                             f"{limit / 60:.0f} min per job (MAX_AUDIO_SECONDS).")
        worker_io.check_base64_fits(s["output"], seconds)

        import torch
        import engine
        pipeline, motion = model()
        portrait = engine.frame_portrait(rgb, getattr(engine, FRAMINGS[s["framing"]]), s["face_ratio"])
        video = os.path.join(work, f"{job_id}.mp4")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = engine.render(pipeline, motion, portrait, torch.from_numpy(waveform), rate, video,
                              anchor=getattr(engine, ANCHORS[s["anchor"]]),
                              reset_seconds=s["reset_seconds"], seed=s["seed"],
                              color_correction=s["color_correction"], on_chunk=Progress(job),
                              parts_dir=os.path.join(work, "parts"), work_dir=work)
        result = worker_io.deliver(video, job_id, s["output"])
        result.update({k: v for k, v in stats.items() if k != "output"})
        result["settings"] = {k: v for k, v in s.items()}
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return result
    except InputError as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def warm_up():
    """torch.compile builds its kernels on the first chunks (a few minutes): do that before
    the worker takes jobs, so no caller pays for it."""
    t = time.time()
    result = handler({"id": "warm-up", "input": {"portrait": "sample", "audio": "sample",
                                                 "max_seconds": 4, "output": "base64"}})
    if "error" in result:
        raise RuntimeError(f"warm-up failed: {result['error']}")
    print(f"[worker] warm-up render done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    import runpod
    model()
    if env_flag("FLASHHEAD_COMPILE"):
        warm_up()
    runpod.serverless.start({"handler": handler})
