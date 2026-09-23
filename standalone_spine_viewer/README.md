# Standalone spine + keypoint viewer

A minimal, self-contained app that shows every detected pose keypoint
(skeleton) and the back-contour spine line, live from a webcam. No
exercise/session logic -- just the raw pose pipeline drawn on screen.

## Setup

```
pip install -r requirements.txt
python main.py
```

Options:
- `--camera N` -- camera index, if your webcam isn't index 0.
- `--backend yolo|mediapipe` -- which pose backend to use (default `yolo`).

Press `f` for fullscreen, `q` or close the window to quit.

## Notes

- Stand **sideways** to the camera (profile view). The spine line is only
  sampled when the person is detected in profile -- it won't appear facing
  the camera head-on.
- Both backends support the spine line, but get the segmentation mask it's
  built from differently:
  - `yolo` (default): re-segments the cropped torso region directly each
    tick, using `yolo11n-pose.pt` + `yolo11n-seg.pt` (both bundled in this
    folder).
  - `mediapipe`: segments the whole frame once per tick instead (it can't
    reliably re-segment an already-cropped torso region in isolation),
    then warps that cached mask into the torso crop's coordinate space.
    Needs the `mediapipe` package.
- This folder is fully self-contained: `core/` here is a standalone copy of
  the main project's pose pipeline, with no dependency on the rest of the
  repo.
