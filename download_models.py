"""Download the FlashHead Pro models into the image at build time (~6.9 GB).

    python download_models.py /models

Layout (what handler.py and the engine expect):
    /models/SoulX-FlashHead-1_3B/Model_Pro/config.json
    /models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors
    /models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth
    /models/wav2vec2-base-960h/{config.json, preprocessor_config.json, model.safetensors}

All three are ungated and Apache-2.0, so no token is needed. Fails loudly if a file is
missing or suspiciously small, so a broken download fails the image build instead of
the first job.
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

FILES = [
    # (repo, file in the repo, folder under /models, minimum size in bytes)
    ("Soul-AILab/SoulX-FlashHead-1_3B", "Model_Pro/config.json", "SoulX-FlashHead-1_3B", 100),
    ("Soul-AILab/SoulX-FlashHead-1_3B", "Model_Pro/diffusion_pytorch_model.safetensors",
     "SoulX-FlashHead-1_3B", 1_000_000_000),
    ("Soul-AILab/SoulX-FlashHead-1_3B", "VAE_Wan/Wan2.1_VAE.pth", "SoulX-FlashHead-1_3B", 100_000_000),
    ("facebook/wav2vec2-base-960h", "config.json", "wav2vec2-base-960h", 100),
    ("facebook/wav2vec2-base-960h", "preprocessor_config.json", "wav2vec2-base-960h", 50),
    ("facebook/wav2vec2-base-960h", "model.safetensors", "wav2vec2-base-960h", 100_000_000),
]


def main(root):
    for repo, name, folder, min_size in FILES:
        path = hf_hub_download(repo, name, local_dir=os.path.join(root, folder))
        size = os.path.getsize(path)
        if size < min_size:
            sys.exit(f"{repo}/{name} is only {size} bytes: the download is broken.")
        print(f"{folder}/{name}: {size / 2**30:.2f} GB", flush=True)
    # hf_hub_download keeps a metadata cache next to the files; the worker never needs it.
    for folder in {f for _, _, f, _ in FILES}:
        cache = os.path.join(root, folder, ".cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/models")
