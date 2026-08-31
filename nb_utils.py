"""
Shared display helpers for the Image-Processing-Techniques notebooks.

Jupyter has no GUI event loop, so cv2.imshow()/cv2.waitKey() either hang or
raise (a headless OpenCV build has no imshow at all). Every notebook in this
repo imports from here instead, so:

  - still images render inline via matplotlib (with the BGR->RGB fix OpenCV
    users always trip over)
  - anything with motion (background subtraction, tracking, optical flow)
    renders as an inline, autoplaying GIF **and** ships a real playable
    HTML5 <video> underneath it, so it works the same in a live Jupyter
    session and in a static GitHub preview of the .ipynb.

Nothing here does anything clever — it exists purely to keep every notebook
free of repeated boilerplate.
"""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, display

REPO_ROOT = Path(__file__).resolve().parent


def _to_rgb(img):
    """BGR (OpenCV) / grayscale -> RGB for correct matplotlib colors."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def show(img, title=None, figsize=(6, 6), cmap=None):
    """Display one BGR/grayscale OpenCV image inline."""
    plt.figure(figsize=figsize)
    rgb = _to_rgb(img)
    if cmap is None and (img.ndim == 2):
        cmap = "gray"
    plt.imshow(rgb, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_grid(images, titles=None, cols=3, figsize=None, cmap=None):
    """Display several BGR/grayscale images side by side for comparison."""
    n = len(images)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    figsize = figsize or (5 * cols, 4.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            c = cmap if cmap is not None else ("gray" if img.ndim == 2 else None)
            ax.imshow(_to_rgb(img), cmap=c)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_gif(frames, path, fps=12, resize_width=None):
    """
    Write a list of BGR frames to an animated GIF.
    `path` is resolved relative to the repo root if not already absolute.
    Returns the resolved Path.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb_frames = []
    for f in frames:
        if resize_width and f.shape[1] != resize_width:
            scale = resize_width / f.shape[1]
            f = cv2.resize(f, (resize_width, int(f.shape[0] * scale)))
        rgb_frames.append(_to_rgb(f))
    imageio.mimsave(str(path), rgb_frames, fps=fps, loop=0)
    return path


def show_gif(path):
    """Embed an animated GIF inline (renders in Jupyter and on GitHub)."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    display(Image(filename=str(path)))


def video_to_gif(video_path, gif_path, fps=10, resize_width=480, max_seconds=None):
    """Read a video file and write it out as an inline-friendly GIF."""
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(src_fps / fps))
    frames = []
    i = 0
    max_frames = int(max_seconds * src_fps) if max_seconds else None
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if i % step == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return save_gif(frames, gif_path, fps=fps, resize_width=resize_width)


def embed_video(path, width=480):
    """
    Embed a real, playable HTML5 <video> inline as a base64 data URI so it
    plays the same in a live Jupyter kernel and in a saved/static .ipynb.
    """
    import base64

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    html = f"""
    <video width="{width}" controls loop muted playsinline>
      <source src="data:video/mp4;base64,{data}" type="video/mp4">
    </video>
    """
    display(HTML(html))


def frames_from_video(path, step=1, max_frames=None):
    """Yield BGR frames from a video file."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cap = cv2.VideoCapture(str(path))
    i = 0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if i % step == 0:
            yield frame
            n += 1
        i += 1
    cap.release()
