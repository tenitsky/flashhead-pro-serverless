# FlashHead Pro: RunPod serverless worker (talking heads from one portrait)

Send **one portrait and a voiceover of any length**; get back a **512 × 512, 25 fps
lip-synced talking-head video** with your original audio. It runs
[SoulX-FlashHead Pro](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul AI Lab, 1.3B,
Apache-2.0) as a RunPod **serverless endpoint**: you pay only while a video renders, and
the endpoint scales to zero when idle.

- About **1.8 s of RTX 5090 time per second of video** (about 2.3 s on an RTX 4090), so
  roughly **$0.04 per finished minute** at RunPod's on-demand prices.
- **6.4 GB peak VRAM**: any 16 GB+ GPU in the endpoint's pools works.
- The models (6.9 GB) are baked into the image; a cold start only loads them.

This is the serverless version of the
[FlashHead Pro pod template](https://github.com/tenitsky/flashhead-pro-avatar) (ComfyUI +
JupyterLab). Same engine, copied into [`engine/`](engine/SOURCE.md); the two never depend
on each other.

## Use it

### 1. Deploy

From the RunPod Hub: open the listing and press **Deploy**. Or by hand: build the image
from this repo (`docker build -t you/flashhead-pro-serverless .`), push it, and create a
serverless endpoint with it (container disk 30 GB, a 16 GB+ GPU).

Then, under the endpoint's **Edit**:

- **Execution timeout**: rendering takes up to ~2.5× the voiceover's length on slower GPUs,
  plus a minute or two. For voiceovers up to 30 min, set **5400 s**.
- **Where the video goes** (pick one, see [Output](#output)): attach a **network volume**,
  or set the `BUCKET_*` variables. Without either, only voiceovers up to 30 s work (the
  video comes back inside the job result).

### 2. Send a job

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"portrait": "https://example.com/me.png",
                 "audio": "https://example.com/voice.mp3"}}'
# -> {"id": "<JOB_ID>", "status": "IN_QUEUE"}

curl https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

While it renders, the status shows progress (`{"chunk": 120, "chunks": 640, "percent": 18.8}`).
Or use the script, which sends local files, polls and saves the mp4:

```bash
pip install requests
python examples/run_job.py --endpoint <ENDPOINT_ID> --portrait me.png --audio voice.mp3
```

Try it with no files of your own: `"portrait": "sample", "audio": "sample"` uses the
portrait and 57 s voiceover bundled with the image.

## Input

| Field | Default | |
|---|---|---|
| `portrait` | required | https URL, base64 (or a `data:` URI), a `/runpod-volume/...` path, or `"sample"`. Any image; the face is found and framed for you |
| `audio` | required | Same forms. Anything ffmpeg reads: mp3, wav, m4a, ogg, even a video's soundtrack |
| `anchor` | `"continuous"` | `continuous`: every 1.12 s chunk continues from the one before (seamless). `refresh_at_pauses`: continuous, but back to the portrait every 40-60 s, in the middle of a real pause (bounds drift on very long videos). `reset_every_n` / `reset_every_chunk`: restart from the portrait on a fixed schedule (for comparison) |
| `reset_seconds` | `40` | Spacing for `refresh_at_pauses` and `reset_every_n` |
| `framing` | `"face"` | `face`: a square 2× the face width, the face centred, as FlashHead was trained. `whole`: center crop of the whole image (for portraits already cropped to the head) |
| `face_ratio` | `2.0` | Square size in face widths (`face` framing) |
| `seed` | `42` | |
| `color_correction` | `1.0` | 0-1: pulls every chunk's colours toward the portrait |
| `max_seconds` | `0` | Render only the first N seconds (0 = all) |
| `output` | `"auto"` | `auto`, `bucket`, `volume` or `base64` |

Inputs larger than 200 MB are refused (`MAX_INPUT_MB`). Bad input never starts a render:
the job ends at once with an `error` that says what to fix.

**Tip:** FlashHead renders the face only (512 × 512 around it). Crop your portrait to a
square around the face yourself if you want to choose the framing.

## Output

| Endpoint has | `output: "auto"` gives | Result field |
|---|---|---|
| `BUCKET_ENDPOINT_URL` (+ keys) | upload to the bucket | `video_url` (a pre-signed link) |
| a network volume | `/runpod-volume/flashhead/outputs/<job id>.mp4` | `video_path` |
| neither | the mp4 inside the result, up to 30 s of video | `video_base64` |

The video runs 6-22 MB per minute, which is why base64 is only for short clips. A longer
voiceover with no bucket or volume is refused **before** rendering (`MAX_BASE64_SECONDS`).

Every result also has `video_mb`, `audio_seconds`, `chunks`, `render_seconds`,
`seconds_per_video_second`, `steady_seconds_per_video_second`, `gpu`, `peak_vram_gb`,
the restarts (`reset_times_s`) and the `settings` used.

**Network volume:** download the videos with RunPod's S3-compatible API for volumes, or
from any pod with the same volume attached. You can also put portraits and voiceovers on
the volume and pass `/runpod-volume/...` paths instead of uploading them with every job.

## Endpoint settings (environment variables)

| Variable | Default | |
|---|---|---|
| `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` | unset | S3-compatible bucket for the videos |
| `FLASHHEAD_COMPILE` | `false` | `true`: torch.compile the model. Faster chunks, but every new worker warms up for a few minutes (with a short sample render) before its first job |
| `MAX_BASE64_SECONDS` | `30` | Longest video returned as base64 |
| `MAX_AUDIO_SECONDS` | `10800` | Longest voiceover per job |
| `MAX_INPUT_MB` | `200` | Largest portrait / voiceover accepted |

## Test locally

With Docker and an NVIDIA GPU (8 GB is enough):

```bash
docker build -t flashhead-pro-serverless:dev .
docker run --rm --gpus all flashhead-pro-serverless:dev \
  python -u handler.py --test_input '{"input": {"portrait": "sample", "audio": "sample", "max_seconds": 5}}'
```

That runs the worker exactly as RunPod does, but the SDK shortens the base64 video when it
logs the result. To get the mp4, call the handler directly and write it to a mounted folder:

```bash
docker run --rm --gpus all -v "$PWD/out:/out" flashhead-pro-serverless:dev python -c "
import base64, json, handler
r = handler.handler({'id': 'test', 'input': {'portrait': 'sample', 'audio': 'sample', 'max_seconds': 5}})
open('/out/test.mp4', 'wb').write(base64.b64decode(r.pop('video_base64')))
print(json.dumps(r, indent=1))"
```

CPU tests of the input, output and job handling (no GPU, no Docker):

```bash
pip install numpy pillow requests pytest
python -m pytest tests -q
```

## How it is built

- [`Dockerfile`](Dockerfile): PyTorch 2.10 + CUDA 12.8 (Blackwell, Ada, Ampere, Hopper),
  ffmpeg, upstream SoulX-FlashHead pinned at `9bc03de`, the packages in
  [`requirements.txt`](requirements.txt) with the image's torch pinned, and the models from
  [`download_models.py`](download_models.py).
- [`handler.py`](handler.py): loads the model once per worker, checks the input, renders
  with the engine, delivers the video.
- [`worker_io.py`](worker_io.py): fetching inputs and delivering outputs.
- [`engine/`](engine/SOURCE.md): the chunk loop, anchors, face framing and ffmpeg output,
  copied from the pod template.
- [`.runpod/`](.runpod): the RunPod Hub listing (`hub.json`) and the test the Hub runs on
  every release (`tests.json`: a 5 s render of the sample on an RTX 4090).

Upstream's `requirements.txt` is not installed: it pins demo and multi-GPU packages
(gradio, xformers, xfuser, mediapipe). On one GPU xfuser and mediapipe are never called;
the engine supplies stand-ins and finds faces with OpenCV's YuNet detector instead.

## What has and hasn't been verified

Verified (October 2026): the image builds from this repo; inside it, on a local RTX 3080
Laptop GPU (8 GB), the worker's own entrypoint (`handler.py --test_input`) loads the model
(about 10 s), takes the job, reports progress and completes, and a direct handler call
returns a playable 512 × 512, 25 fps H.264 + AAC mp4 of exactly the requested 5 s, with
peak VRAM 6.15 GB. The input, output and job handling pass the CPU tests in `tests/`.
The engine itself is the pod template's, whose renders on RunPod measured ~1.8 s per
video second on an RTX 5090 and ~2.3 s on an RTX 4090.

Not yet verified: a deployed RunPod endpoint (cold start from the registry, the bucket
upload, a network volume), `FLASHHEAD_COMPILE=true` in this image, and the Hub's own build
and test run. An 8 GB laptop GPU renders far slower (20-40 s per video second) than any
GPU in the endpoint's pools; its timing says nothing about a deployed endpoint.

## Models

| | Size | Licence |
|---|---|---|
| [SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) `Model_Pro` | 6.0 GB | Apache-2.0 |
| Wan 2.1 VAE (same repo) | 0.5 GB | Apache-2.0 |
| [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | 0.4 GB | Apache-2.0 |

## Credits

[SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul AI Lab,
[technical report](https://arxiv.org/pdf/2602.07449)) ·
[YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
(OpenCV Zoo, MIT) · [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
(Meta, Apache-2.0) · built by [AI Art Pipeline](https://www.youtube.com/@aiartpipeline).
