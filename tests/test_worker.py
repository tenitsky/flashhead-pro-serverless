"""CPU tests for the worker: input checks, fetching, output modes and the handler flow.

    python -m pytest tests -q

The GPU parts (torch, the FlashHead engine, runpod) are replaced with small fakes, so
this runs anywhere with numpy, Pillow, requests and ffmpeg. The real render is checked
with the Docker image instead (README, "Test locally").
"""
import base64
import importlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SAMPLE_PORTRAIT = os.path.join(ROOT, "samples", "sample_portrait.png")
SAMPLE_AUDIO = os.path.join(ROOT, "samples", "sample_voiceover.mp3")


# ---------------------------------------------------------------------------- fakes
class FakeTensor:
    def __init__(self, array):
        self.array = array


def fake_torch():
    torch = types.ModuleType("torch")
    torch.from_numpy = FakeTensor
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    return torch


def fake_engine(calls):
    engine = types.ModuleType("engine")
    for name, value in (("ANCHOR_CONTINUOUS", "continuous (upstream)"),
                        ("ANCHOR_PAUSES", "refresh from portrait at a pause"),
                        ("ANCHOR_EVERY_CHUNK", "reset to portrait every chunk"),
                        ("ANCHOR_EVERY_N", "reset to portrait every N seconds"),
                        ("FRAMING_FACE", "face crop"), ("FRAMING_WHOLE", "whole image (center crop)")):
        setattr(engine, name, value)

    def load_pipeline(ckpt, w2v, model_type="pro", compile_model=True):
        calls["load"] = calls.get("load", 0) + 1
        return "pipeline", 5

    def frame_portrait(rgb, framing, ratio):
        calls["framing"] = (framing, ratio, rgb.shape)
        return np.zeros((512, 512, 3), np.uint8)

    def render(pipeline, motion, portrait, waveform, rate, out, **kw):
        calls["render"] = dict(kw, rate=rate, samples=waveform.array.shape[1])
        for i in range(3):
            kw["on_chunk"](i, 3, None)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                        "-pix_fmt", "yuv420p", out], check=True)
        return {"output": out, "chunks": 3, "audio_seconds": waveform.array.shape[1] / rate,
                "render_seconds": 1.0, "seconds_per_video_second": 0.5}

    engine.load_pipeline, engine.frame_portrait, engine.render = load_pipeline, frame_portrait, render
    return engine


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """handler + worker_io with fake torch / engine / runpod; no volume, no bucket."""
    calls, progress = {}, []
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(progress_update=lambda job, p: progress.append(p))
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "engine", fake_engine(calls))
    monkeypatch.setitem(sys.modules, "runpod", runpod)
    monkeypatch.delenv("BUCKET_ENDPOINT_URL", raising=False)
    import worker_io
    import handler
    importlib.reload(worker_io)
    importlib.reload(handler)
    monkeypatch.setattr(worker_io, "VOLUME", str(tmp_path / "no-volume"))
    monkeypatch.setattr(handler, "PROGRESS_EVERY", 0.0)
    return types.SimpleNamespace(handler=handler, io=worker_io, calls=calls, progress=progress,
                                 tmp=tmp_path)


def run(worker, **inp):
    return worker.handler.handler({"id": "job-1", "input": inp})


# ---------------------------------------------------------------------------- inputs
def test_sample_files_exist():
    assert os.path.getsize(SAMPLE_PORTRAIT) > 100_000
    assert os.path.getsize(SAMPLE_AUDIO) > 100_000


def test_fetch_sample_base64_and_data_uri(worker):
    io = worker.io
    assert io.fetch("sample", "portrait", str(worker.tmp)) == io.SAMPLES["portrait"]
    raw = b"hello"
    path = io.fetch(base64.b64encode(raw).decode(), "audio", str(worker.tmp))
    assert open(path, "rb").read() == raw
    path = io.fetch("data:audio/mpeg;base64," + base64.b64encode(raw).decode(), "audio", str(worker.tmp))
    assert open(path, "rb").read() == raw


@pytest.mark.parametrize("value", [None, "", "   ", 42, "not base64 !!", "/etc/passwd"])
def test_fetch_rejects_bad_values(worker, value):
    with pytest.raises(worker.io.InputError):
        worker.io.fetch(value, "portrait", str(worker.tmp))


def test_fetch_volume_path_stays_on_volume(worker, monkeypatch):
    volume = worker.tmp / "vol"
    (volume / "in").mkdir(parents=True)
    (volume / "in" / "a.mp3").write_bytes(b"x")
    monkeypatch.setattr(worker.io, "VOLUME", str(volume))
    assert worker.io.fetch(f"{volume}/in/a.mp3", "audio", str(worker.tmp)).endswith("a.mp3")
    with pytest.raises(worker.io.InputError):
        worker.io.fetch(f"{volume}/../outside.mp3", "audio", str(worker.tmp))
    with pytest.raises(worker.io.InputError):
        worker.io.fetch(f"{volume}/in/missing.mp3", "audio", str(worker.tmp))


def test_fetch_size_limit(worker, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_MB", "0.000001")
    with pytest.raises(worker.io.InputError, match="limit"):
        worker.io.fetch(base64.b64encode(b"x" * 100).decode(), "audio", str(worker.tmp))


def test_download_errors_become_input_errors(worker, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(worker.io.InputError, match="Could not download"):
        worker.io.fetch("https://example.invalid/a.png", "portrait", str(worker.tmp))


def test_load_audio_and_image(worker):
    wave, rate = worker.io.load_audio(SAMPLE_AUDIO)
    assert rate == 48000 and wave.shape[0] == 2 and wave.dtype == np.float32
    assert 55 < wave.shape[1] / rate < 60
    rgb = worker.io.load_image(SAMPLE_PORTRAIT)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    bad = worker.tmp / "bad"
    bad.write_bytes(b"not media")
    with pytest.raises(worker.io.InputError):
        worker.io.load_image(str(bad))
    with pytest.raises(worker.io.InputError):
        worker.io.load_audio(str(bad))


# ---------------------------------------------------------------------------- outputs
def test_resolve_mode(worker, monkeypatch):
    io = worker.io
    assert io.resolve_mode(None) == "base64"
    with pytest.raises(io.InputError):
        io.resolve_mode("bucket")
    with pytest.raises(io.InputError):
        io.resolve_mode("volume")
    with pytest.raises(io.InputError):
        io.resolve_mode("email")
    (worker.tmp / "vol").mkdir()
    monkeypatch.setattr(io, "VOLUME", str(worker.tmp / "vol"))
    assert io.resolve_mode("auto") == "volume"
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://s3.example")
    assert io.resolve_mode("auto") == "bucket"


def test_base64_length_guard(worker):
    worker.io.check_base64_fits("base64", 30)
    worker.io.check_base64_fits("volume", 3600)
    with pytest.raises(worker.io.InputError, match="network volume"):
        worker.io.check_base64_fits("base64", 31)


# ---------------------------------------------------------------------------- handler
def test_handler_renders_sample_as_base64(worker):
    result = run(worker, portrait="sample", audio="sample", max_seconds=5, seed=7)
    assert "error" not in result, result
    assert base64.b64decode(result["video_base64"])[4:8] == b"ftyp"
    assert result["chunks"] == 3 and "output" not in result
    render = worker.calls["render"]
    assert render["seed"] == 7 and render["anchor"] == "continuous (upstream)"
    assert render["samples"] == 5 * 48000
    assert worker.calls["framing"][0] == "face crop" and worker.calls["framing"][1] == 2.0
    assert worker.progress[-1] == {"chunk": 3, "chunks": 3, "percent": 100.0}


def test_handler_saves_to_volume(worker, monkeypatch):
    volume = worker.tmp / "vol"
    volume.mkdir()
    monkeypatch.setattr(worker.io, "VOLUME", str(volume))
    result = run(worker, portrait="sample", audio="sample", anchor="refresh_at_pauses", framing="whole")
    assert result["video_path"] == str(volume / "flashhead" / "outputs" / "job-1.mp4")
    assert os.path.getsize(result["video_path"]) > 0
    assert worker.calls["render"]["anchor"] == "refresh from portrait at a pause"
    assert worker.calls["framing"][0] == "whole image (center crop)"


def test_handler_refuses_long_base64_before_rendering(worker):
    result = run(worker, portrait="sample", audio="sample")       # 57 s, no volume or bucket
    assert "MAX_BASE64_SECONDS" in result["error"]
    assert "render" not in worker.calls and "load" not in worker.calls


@pytest.mark.parametrize("inp", [
    {"audio": "sample"},
    {"portrait": "sample", "audio": "sample", "anchor": "sometimes"},
    {"portrait": "sample", "audio": "sample", "seed": "abc"},
    {"portrait": "sample", "audio": "sample", "color_correction": 2},
    {"portrait": "sample", "audio": "sample", "output": "bucket"},
])
def test_handler_input_errors(worker, inp):
    result = run(worker, **inp)
    assert set(result) == {"error"}
    assert "render" not in worker.calls


def test_handler_cleans_its_temp_folder(worker, monkeypatch):
    made = []
    real = worker.handler.tempfile.mkdtemp

    def track(**kw):
        made.append(real(**kw))
        return made[-1]
    monkeypatch.setattr(worker.handler.tempfile, "mkdtemp", track)
    run(worker, portrait=base64.b64encode(open(SAMPLE_PORTRAIT, "rb").read()).decode(),
        audio="sample", max_seconds=2)
    run(worker, portrait="sample", audio="sample", anchor="bad")
    assert made and not any(os.path.exists(p) for p in made)
