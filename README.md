# Image-Processing-Techniques
Image Processing using OpenCV

A 26-notebook walkthrough of classical computer vision with OpenCV — from
loading your first image to background subtraction, optical flow, and object
tracking. Every notebook runs top to bottom and renders its results inline:
static outputs as embedded plots, motion outputs as autoplaying GIFs *and*
real scrubbable video players, no local GUI or download step required.

## What's here

| # | Notebook | Covers |
|---|----------|--------|
| 01 | Getting started with images | imread/imshow/imwrite, image shape |
| 02 | Grayscaling in Images | `cvtColor` to grayscale |
| 03 | Color Spaces | BGR channels, HSV |
| 04 | Drawing on images | lines, rectangles, circles, polygons, text |
| 05 | Transformations | translation, rotation, flipping |
| 06 | Scaling, Resizing & Cropping | interpolation methods, image pyramids |
| 07 | Arithmetic & Bitwise Operations | add/subtract, AND/OR/XOR/NOT, masking |
| 08 | Convolutions, Blurring & Sharpening | kernels, denoising |
| 09 | Thresholding & Binarization | global, adaptive, Otsu, scikit-image local |
| 10 | Dilation, Erosion & Edge Detection | morphology, Canny, auto-Canny |
| 11 | Contours | `findContours` retrieval modes & approximation |
| 12 | Sorting & Matching Contours | moments, `approxPolyDP`, convex hull, `matchShapes` |
| 13 | Line, Circle & Blob Detection | Hough lines/circles, `SimpleBlobDetector` |
| 14 | Counting Circles & Template Matching | blob filtering, `matchTemplate` |
| 15 | Finding Corners | Harris, `goodFeaturesToTrack` |
| 16 | Face & Eye Detection | Haar cascades, webcam capture |
| 17 | Vehicle & Pedestrian Detection | Haar cascade + background-subtraction detection on video |
| 18 | Perspective Transforms | `getPerspectiveTransform`, document unwarping |
| 19 | Histograms & K-Means Color Analysis | channel histograms, dominant-color clustering |
| 20 | Comparing Images | MSE, structural similarity |
| 21 | Filtering Colors | HSV color-range masking |
| 22 | Watershed Segmentation | marker-based segmentation of touching objects |
| 23 | Background Subtraction | MOG2, KNN, running-average foreground masks |
| 24 | Motion Tracking | meanshift, CAMSHIFT |
| 25 | Object Tracking with Optical Flow | Lucas-Kanade (sparse), Farneback (dense) |
| 26 | Single Object Tracking by Color | HSV threshold + contour tracking |

Every lesson closes with a short, plainly-worded section of a few concrete,
runnable extensions of that notebook's technique — not just a summary of
what was covered, and not the same stamped heading repeated 26 times.

Notebooks whose technique has real math worth unpacking also get an
intuitive, from-scratch explanation of *why* it works, right where the
concept first appears — not just a link to a paper. A few get a small
generated diagram because a picture genuinely beats prose there: interpolation
methods made visible by zooming into a tiny image (06), a convolution kernel
sliding over actual numbers (08), the Hough-transform "points become
sinusoids, collinear points' sinusoids cross at one point" trick (13), the
Harris corner flat/edge/corner eigenvalue plane (15), and a worked integral-image
example (16). Others (Otsu's threshold, image moments, homographies, K-means,
MSE vs. SSIM, watershed's distance transform, Gaussian-mixture background
models, meanshift/CAMSHIFT, the Lucas-Kanade optical flow equation) get a
tight written explanation in place.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook
```

That's it — `assets/images/`, `assets/videos/`, and `assets/haarcascades/`
ship with the repo, so nothing needs to be downloaded before running a
notebook top to bottom.

`opencv-python-headless` is pinned in `requirements.txt` to a version known
to include `cv2.CascadeClassifier` — the 5.0.0 headless wheel available at
the time of writing shipped without it.

## A note on the sample assets

The original course this repo is based on distributed its images, video
clips, and Haar cascade files separately from the notebooks (a `!wget` /
`!unzip` step against links that have since gone stale) — none of it was
ever committed to git here, in any past commit. Every notebook still expects
those files at their original relative names, so `assets/images/`,
`assets/videos/`, and `assets/haarcascades/Haarcascades/` in this repo are a
real, working replacement set, wired up under (mostly) those same names —
all consolidated under a single top-level `assets/` folder rather than
scattered loose at the repo root:

- **Real photos/scenes** — sourced from **[OpenCV's own official sample
  data](https://github.com/opencv/opencv/tree/4.x/samples/data)**
  (`fruits.jpg`, `messi5.jpg`, `lena.jpg`, `sudoku.png`, `cards.png`,
  `left01.jpg`, `licenseplate_motion.jpg`, `ela_original/modified.jpg`, and
  more) and **[scikit-image](https://scikit-image.org/)**'s bundled samples
  (`coins`, `page`) — both permissively licensed and exactly the kind of
  imagery these lessons were originally written against.
- **Haar cascades** — the official `haarcascade_frontalface_default.xml`,
  `haarcascade_eye.xml`, and `haarcascade_fullbody.xml` from
  [opencv/opencv](https://github.com/opencv/opencv/tree/4.x/data/haarcascades).
  There's no officially-maintained, clearly-licensed `haarcascade_car.xml` —
  the copies that circulate in tutorials trace back to an old forum post
  with no formal license — so notebook 17's vehicle detection uses
  background subtraction instead (see the note in that notebook).
- **Procedurally generated** — where no suitable real sample existed:
  the contour-sorting shape set (`4star`, `bunchofshapes`, `hand`, `house`,
  `shapestomatch`), the Hough-circle scene, and the skewed "scanned
  document" for perspective transforms — all built by
  `scripts/generate_assets.py`, so they're fully reproducible from code.
  Notebook 14's "find the target in a busy scene" demo is a from-scratch
  template-matching exercise, not the original book's Where's Waldo art
  (which isn't ours to redistribute).
- **`assets/videos/walking.avi` / `assets/videos/walking_short_clip.mp4`** —
  a trimmed, downscaled clip of OpenCV's own `vtest.avi` sample (people
  crossing a plaza), used for pedestrian detection, background subtraction,
  and optical flow.
- **Real vehicle footage** (notebooks 17, 24, 26) — `assets/videos/cars.mp4`
  is a real multi-lane highway clip from
  [Pixabay](https://pixabay.com/videos/highway-traffic-vehicles-cars-road-56310/)
  (Pixabay Content License — free, no attribution required), used for
  notebook 17's background-subtraction vehicle detector, where real traffic
  makes for a better multi-object stress test than one synthetic car ever
  could. `data_slow.flv` / `car-detection.mp4` are the same
  `slow_traffic_small.mp4` clip OpenCV's own official meanshift/CAMSHIFT
  tutorials use (see `samples/python/tutorial_code/video/meanshift/` in
  [opencv/opencv](https://github.com/opencv/opencv)) — one clearly dominant
  car, which is exactly what single-object tracking (24, 26) needs. It's
  hosted on a third-party tutorial site with no formally stated license, not
  OpenCV's own repo — flagged here rather than glossed over. Both are
  reproducible via `scripts/fetch_traffic_videos.sh`.

A couple of notebooks needed a small, called-out code fix alongside the
asset swap — a real Hough-line shape-compatibility issue across OpenCV
versions, a `np.int0` removal in NumPy 2.0, and a genuine pre-existing bug in
the meanshift/CAMSHIFT setup (the tracked-object histogram was being built
from the whole frame instead of the crop) that only surfaced once the
notebook was actually run against real output. Each is commented in place.

## How the inline video/GIF rendering works

`nb_utils.py` (imported by the video-heavy notebooks) provides:

- `show()` / `show_grid()` — matplotlib display with the BGR→RGB fix,
  single image or side-by-side comparison.
- `save_gif()` / `video_to_gif()` — turns a list of frames (or an existing
  video file) into an animated GIF, embedded directly in the notebook so it
  renders in a live Jupyter session *and* in a static GitHub preview.
- `embed_video()` — base64-embeds an actual `<video>` player inline (real
  scrub/pause controls) for when you're running the notebook live.

`cv2.imshow()` / `cv2.waitKey()` never work inside a notebook — there's no
GUI event loop — so every place the original code relied on them (or wrote a
video file and never displayed it at all) now goes through one of the above
instead.

## Develop

```bash
# Regenerate the procedurally-generated assets
.venv/bin/python scripts/generate_assets.py

# Rebuild + re-execute a notebook (pass its number, or nothing for all 26)
JUPYTER_DATA_DIR="$(pwd)/.venv/share/jupyter" .venv/bin/python scripts/build_notebooks.py 17
```

`scripts/build_notebooks.py` is idempotent only against a freshly-checked-out
notebook — it edits specific cells by matching their original source text, so
re-running it against its own output will fail to find what it's looking
for. `git checkout -- "<notebook>.ipynb"` before re-running.
