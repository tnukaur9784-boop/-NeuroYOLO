# How to Use `gait_pipeline.py` in Google Colab

This guide shows a **reproducible Colab workflow** for running markerless gait analysis on one walking video.

## 1) Open Colab and install dependencies

Create a new notebook and run:

```python
!pip install -q opencv-python mediapipe numpy pandas scipy matplotlib
```

---

## 2) Upload your walking video

```python
from google.colab import files
uploaded = files.upload()
video_name = list(uploaded.keys())[0]
print("Uploaded:", video_name)
```

---

## 3) Add the analyzer script

### Option A (recommended): Upload `gait_pipeline.py`

```python
uploaded = files.upload()  # choose gait_pipeline.py from your computer
```

### Option B: Clone your repo

```python
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_FOLDER>
```

---

## 4) Run gait analysis

Use either a known scale or subject height.

### A. If you know scale (meters-per-pixel)

```python
!python gait_pipeline.py \
  --video "$video_name" \
  --scale_m_per_px 0.0025 \
  --out_csv gait_metrics.csv \
  --out_plot gait_diagnostics.png
```

### B. If you do **not** know scale, use subject height

```python
!python gait_pipeline.py \
  --video "$video_name" \
  --subject_height_m 1.68 \
  --out_csv gait_metrics.csv \
  --out_plot gait_diagnostics.png
```

---

## 5) View and download outputs

```python
import pandas as pd
from IPython.display import Image, display
from google.colab import files

# Inspect metrics table
metrics = pd.read_csv("gait_metrics.csv")
print(metrics.to_string(index=False))

# Show diagnostics figure
display(Image("gait_diagnostics.png"))

# Download files
files.download("gait_metrics.csv")
files.download("gait_diagnostics.png")
```

---

## 6) Recommended settings if step count still looks off

Try one change at a time:

```python
# less noise / more stable event signals
!python gait_pipeline.py --video "$video_name" --subject_height_m 1.68 --smoothing butter

# reduce dropped detail from skipping frames (default is already 1)
!python gait_pipeline.py --video "$video_name" --subject_height_m 1.68 --frame_skip 1
```

Also ensure:
- Full body is visible throughout gait trial.
- Camera is as stable as possible.
- Walking segment is trimmed to only the valid gait interval.

---

## 7) Clinical sanity checks (quick)

Before trusting outputs, visually confirm in `gait_diagnostics.png`:
- Heel-strike markers appear near alternating left/right contact moments.
- Step interval series does not contain many extreme outliers.
- Cadence and speed match rough stopwatch/manual estimates.

If your manually counted steps are around **28–36**, compare that with the CSV `Total Steps (steps)` and re-run after trimming non-walking portions.
