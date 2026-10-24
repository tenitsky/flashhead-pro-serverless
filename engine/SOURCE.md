# Where `engine/` comes from

`engine.py` and `face_detection_yunet_2023mar.onnx` are unmodified copies from the
FlashHead Pro **pod template**,
[tenitsky/flashhead-pro-avatar](https://github.com/tenitsky/flashhead-pro-avatar),
commit `6c3bbe6` (`custom_nodes/comfyui-flashhead/`). The engine has no ComfyUI imports:
it drives upstream SoulX-FlashHead's own pipeline code, chunk by chunk, and writes the mp4
with ffmpeg.

They are copied, not fetched, so this worker and the pod template never break each other.
To take a newer engine, copy both files again from a commit of that repo and update the
commit above.

- `engine.py`: chunk loop, anchors (continuous / refresh at pauses / resets), face framing,
  parts, ffmpeg output.
- `face_detection_yunet_2023mar.onnx`: OpenCV Zoo's YuNet face detector (MIT licence),
  used to frame the portrait the way FlashHead was trained.
