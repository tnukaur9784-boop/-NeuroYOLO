"""Robust monocular gait analysis pipeline for clinical research."""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

warnings.filterwarnings("ignore", category=UserWarning)

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class GaitConfig:
    frame_skip: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    smoothing: str = "savgol"
    savgol_window: int = 11
    savgol_polyorder: int = 3
    butter_cutoff_hz: float = 4.0
    butter_order: int = 3
    min_step_time_s: float = 0.22
    max_step_time_s: float = 1.30
    min_stride_time_s: float = 0.55
    max_stride_time_s: float = 3.0


@dataclass
class EventSeries:
    heel_strike_idx: np.ndarray
    toe_off_idx: np.ndarray


@dataclass
class AnalysisOutput:
    metrics: Dict[str, float]
    frame_table: pd.DataFrame
    events: Dict[str, EventSeries]
    diagnostics: Dict[str, np.ndarray]


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _safe_cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    mu = float(np.mean(arr))
    if abs(mu) < 1e-8:
        return 0.0
    return float(np.std(arr, ddof=1) / mu * 100.0)


def _regularity_index(step_intervals: Sequence[float]) -> float:
    arr = np.asarray(step_intervals, dtype=float)
    if arr.size < 4:
        return 0.0
    x, y = arr[:-1], arr[1:]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(np.clip(r, 0.0, 1.0) * 100.0)


class MarkerlessGaitAnalyzer:
    def __init__(
        self,
        video_path: Path,
        scale_m_per_px: Optional[float] = None,
        subject_height_m: Optional[float] = None,
        config: Optional[GaitConfig] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.scale_m_per_px = scale_m_per_px
        self.subject_height_m = subject_height_m
        self.config = config or GaitConfig()
        self.fps = 0.0

    def _extract_pose(self) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        rows: List[Dict[str, float]] = []
        frame_idx = -1
        proc_idx = 0
        tracked = [
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_ANKLE,
            RIGHT_ANKLE,
            LEFT_HEEL,
            RIGHT_HEEL,
            LEFT_FOOT_INDEX,
            RIGHT_FOOT_INDEX,
        ]

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % self.config.frame_skip != 0:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                row = {
                    "frame": frame_idx,
                    "t": proc_idx / (self.fps / self.config.frame_skip),
                    "w": w,
                    "h": h,
                }
                for i in tracked:
                    row[f"x_{i}"] = np.nan
                    row[f"y_{i}"] = np.nan

                if res.pose_landmarks:
                    for i in tracked:
                        lm = res.pose_landmarks.landmark[i]
                        row[f"x_{i}"] = lm.x * w
                        row[f"y_{i}"] = lm.y * h

                rows.append(row)
                proc_idx += 1
        finally:
            pose.close()
            cap.release()

        if not rows:
            raise RuntimeError("No frames could be processed.")

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col.startswith(("x_", "y_")):
                df[col] = df[col].interpolate(limit_direction="both")
        return df

    def _smooth(self, x: np.ndarray, fs: float) -> np.ndarray:
        if len(x) < 7:
            return x.copy()
        if self.config.smoothing == "butter":
            nyq = fs / 2.0
            cutoff = np.clip(self.config.butter_cutoff_hz / nyq, 0.01, 0.99)
            b, a = butter(self.config.butter_order, cutoff, btype="low")
            return filtfilt(b, a, x)

        win = min(self.config.savgol_window, len(x) - (1 - len(x) % 2))
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)
        poly = min(self.config.savgol_polyorder, win - 1)
        return savgol_filter(x, window_length=win, polyorder=poly, mode="interp")

    def _derive_scale(self, df: pd.DataFrame) -> float:
        if self.scale_m_per_px is not None and self.scale_m_per_px > 0:
            return float(self.scale_m_per_px)
        if self.subject_height_m is None:
            raise ValueError("Provide --scale_m_per_px or --subject_height_m.")

        hip_y = 0.5 * (df[f"y_{LEFT_HIP}"] + df[f"y_{RIGHT_HIP}"])
        ank_y = 0.5 * (df[f"y_{LEFT_ANKLE}"] + df[f"y_{RIGHT_ANKLE}"])
        leg_px = np.nanmedian(np.abs(ank_y - hip_y))
        if not np.isfinite(leg_px) or leg_px <= 1:
            raise ValueError("Unable to estimate anthropometric scale from hip/ankle distance.")
        return float(0.53 * self.subject_height_m / leg_px)

    def _build_signals(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        fs = self.fps / self.config.frame_skip

        pelvis = np.column_stack(
            [
                0.5 * (df[f"x_{LEFT_HIP}"].to_numpy() + df[f"x_{RIGHT_HIP}"].to_numpy()),
                0.5 * (df[f"y_{LEFT_HIP}"].to_numpy() + df[f"y_{RIGHT_HIP}"].to_numpy()),
            ]
        )
        l_heel = np.column_stack([df[f"x_{LEFT_HEEL}"].to_numpy(), df[f"y_{LEFT_HEEL}"].to_numpy()])
        r_heel = np.column_stack([df[f"x_{RIGHT_HEEL}"].to_numpy(), df[f"y_{RIGHT_HEEL}"].to_numpy()])

        # Data-driven progression axis (for frontal/side/oblique).
        heel_rel = np.vstack([l_heel - pelvis, r_heel - pelvis])
        heel_rel = heel_rel - np.mean(heel_rel, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(heel_rel, full_matrices=False)
        prog_axis = vt[0]

        # Orient progression axis toward global pelvis displacement.
        if np.dot(pelvis[-1] - pelvis[0], prog_axis) < 0:
            prog_axis = -prog_axis

        pelvis_prog = np.dot(pelvis, prog_axis)
        l_heel_prog = np.dot(l_heel, prog_axis)
        r_heel_prog = np.dot(r_heel, prog_axis)

        rel_l = l_heel_prog - pelvis_prog
        rel_r = r_heel_prog - pelvis_prog

        sig = {
            "t": df["t"].to_numpy(),
            "pelvis_prog": self._smooth(pelvis_prog, fs),
            "l_heel_prog": self._smooth(l_heel_prog, fs),
            "r_heel_prog": self._smooth(r_heel_prog, fs),
            "rel_l": self._smooth(rel_l, fs),
            "rel_r": self._smooth(rel_r, fs),
            "ankle_y_l": self._smooth(df[f"y_{LEFT_ANKLE}"].to_numpy(), fs),
            "ankle_y_r": self._smooth(df[f"y_{RIGHT_ANKLE}"].to_numpy(), fs),
        }
        sig["heel_gap_prog"] = sig["r_heel_prog"] - sig["l_heel_prog"]
        return sig

    def _detect_hs_from_gap(self, gap: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Detect right HS as maxima and left HS as minima of inter-heel progression gap."""
        min_step_dist = max(1, int(self.config.min_step_time_s * fs * 0.8))
        prominence = max(np.std(gap) * 0.12, 0.8)

        r_hs, _ = find_peaks(gap, distance=min_step_dist, prominence=prominence)
        l_hs, _ = find_peaks(-gap, distance=min_step_dist, prominence=prominence)

        return np.sort(l_hs.astype(int)), np.sort(r_hs.astype(int))

    def _filter_stride_intervals(self, hs: np.ndarray, fs: float) -> np.ndarray:
        if hs.size < 2:
            return hs
        keep = [0]
        for i in range(1, hs.size):
            dt = (hs[i] - hs[i - 1]) / fs
            if self.config.min_stride_time_s <= dt <= self.config.max_stride_time_s:
                keep.append(i)
        return hs[np.asarray(keep, dtype=int)]

    def _enforce_alternation(self, l_hs: np.ndarray, r_hs: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove impossible same-side repeats too close in time after initial detection."""
        merged = sorted([(i, "L") for i in l_hs] + [(i, "R") for i in r_hs], key=lambda x: x[0])
        if not merged:
            return l_hs, r_hs

        filtered = [merged[0]]
        for idx, side in merged[1:]:
            prev_idx, prev_side = filtered[-1]
            dt = (idx - prev_idx) / fs
            if side == prev_side and dt < self.config.min_step_time_s:
                continue
            filtered.append((idx, side))

        l_new = np.array([i for i, s in filtered if s == "L"], dtype=int)
        r_new = np.array([i for i, s in filtered if s == "R"], dtype=int)
        return l_new, r_new

    def _detect_to_between_hs(self, hs: np.ndarray, rel: np.ndarray) -> np.ndarray:
        if hs.size < 2:
            return np.array([], dtype=int)
        to_idx: List[int] = []
        for a, b in zip(hs[:-1], hs[1:]):
            if b <= a + 2:
                continue
            seg = rel[a:b]
            if seg.size < 3:
                continue
            k = int(np.argmin(seg))
            cand = a + k
            if a < cand < b:
                to_idx.append(cand)
        return np.asarray(to_idx, dtype=int)

    def _events(self, sig: Dict[str, np.ndarray]) -> Dict[str, EventSeries]:
        fs = self.fps / self.config.frame_skip
        l_hs, r_hs = self._detect_hs_from_gap(sig["heel_gap_prog"], fs)
        l_hs, r_hs = self._enforce_alternation(l_hs, r_hs, fs)
        l_hs = self._filter_stride_intervals(l_hs, fs)
        r_hs = self._filter_stride_intervals(r_hs, fs)

        l_to = self._detect_to_between_hs(l_hs, sig["rel_l"])
        r_to = self._detect_to_between_hs(r_hs, sig["rel_r"])
        return {
            "left": EventSeries(l_hs, l_to),
            "right": EventSeries(r_hs, r_to),
        }

    def _temporal(self, events: Dict[str, EventSeries], fs: float) -> Dict[str, List[float]]:
        l_hs = events["left"].heel_strike_idx
        r_hs = events["right"].heel_strike_idx
        l_to = events["left"].toe_off_idx
        r_to = events["right"].toe_off_idx

        l_stride = np.diff(l_hs) / fs if l_hs.size >= 2 else np.array([])
        r_stride = np.diff(r_hs) / fs if r_hs.size >= 2 else np.array([])

        merged = sorted([(i, "L") for i in l_hs] + [(i, "R") for i in r_hs], key=lambda x: x[0])
        step_times: List[float] = []
        step_l: List[float] = []
        step_r: List[float] = []
        for (i0, s0), (i1, s1) in zip(merged[:-1], merged[1:]):
            if s0 == s1:
                continue
            dt = (i1 - i0) / fs
            if self.config.min_step_time_s <= dt <= self.config.max_step_time_s:
                step_times.append(dt)
                if s1 == "L":
                    step_l.append(dt)
                else:
                    step_r.append(dt)

        stance_l, swing_l = [], []
        for hs, to in zip(l_hs[:-1], l_to):
            if hs < to:
                stance_l.append((to - hs) / fs)
        for to, hs_n in zip(l_to, l_hs[1:]):
            if to < hs_n:
                swing_l.append((hs_n - to) / fs)

        stance_r, swing_r = [], []
        for hs, to in zip(r_hs[:-1], r_to):
            if hs < to:
                stance_r.append((to - hs) / fs)
        for to, hs_n in zip(r_to, r_hs[1:]):
            if to < hs_n:
                swing_r.append((hs_n - to) / fs)

        ds = []
        for hs_l in l_hs:
            after = r_to[r_to > hs_l]
            if after.size:
                x = (after[0] - hs_l) / fs
                if 0 < x < 0.8:
                    ds.append(x)
        for hs_r in r_hs:
            after = l_to[l_to > hs_r]
            if after.size:
                x = (after[0] - hs_r) / fs
                if 0 < x < 0.8:
                    ds.append(x)

        return {
            "step_times": step_times,
            "step_times_left": step_l,
            "step_times_right": step_r,
            "stride_times_left": l_stride.tolist(),
            "stride_times_right": r_stride.tolist(),
            "stance_times": stance_l + stance_r,
            "swing_times": swing_l + swing_r,
            "cycle_times": np.concatenate([l_stride, r_stride]).tolist() if (l_stride.size + r_stride.size) else [],
            "double_support": ds,
        }

    def _spatial(self, events: Dict[str, EventSeries], sig: Dict[str, np.ndarray], scale: float) -> Dict[str, List[float] | float]:
        l_hs = events["left"].heel_strike_idx
        r_hs = events["right"].heel_strike_idx
        merged = sorted([(i, "L") for i in l_hs] + [(i, "R") for i in r_hs], key=lambda x: x[0])

        step_lengths, step_l, step_r = [], [], []
        for idx, side in merged:
            d = abs(sig["r_heel_prog"][idx] - sig["l_heel_prog"][idx]) * scale
            if 0.05 <= d <= 2.0:
                step_lengths.append(float(d))
                if side == "L":
                    step_l.append(float(d))
                else:
                    step_r.append(float(d))

        # Estimate stride length as sum of two consecutive step lengths.
        stride_lengths = []
        for a, b in zip(step_lengths[:-1], step_lengths[1:]):
            s = a + b
            if 0.2 <= s <= 4.0:
                stride_lengths.append(float(s))

        return {
            "step_lengths": step_lengths,
            "step_lengths_left": step_l,
            "step_lengths_right": step_r,
            "stride_lengths": stride_lengths,
            "mean_step_length": _safe_mean(step_lengths),
            "mean_stride_length": _safe_mean(stride_lengths),
        }

    def analyze(self) -> AnalysisOutput:
        df = self._extract_pose()
        fs = self.fps / self.config.frame_skip
        scale = self._derive_scale(df)
        sig = self._build_signals(df)
        events = self._events(sig)

        temporal = self._temporal(events, fs)
        spatial = self._spatial(events, sig, scale)

        duration = float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0
        total_steps = len(temporal["step_times"])
        cadence = (total_steps / duration) * 60.0 if duration > 0 else 0.0
        speed = (spatial["mean_step_length"] * cadence / 60.0) if cadence > 0 else 0.0

        cycle_time = _safe_mean(temporal["cycle_times"])
        ds_time = _safe_mean(temporal["double_support"])
        ds_ratio = (ds_time / cycle_time * 100.0) if cycle_time > 0 else 0.0

        mean_step_r = _safe_mean(spatial["step_lengths_right"])
        mean_step_l = _safe_mean(spatial["step_lengths_left"])
        step_asym = abs(mean_step_r - mean_step_l) / max((mean_step_r + mean_step_l) / 2.0, 1e-6) * 100.0

        metrics = {
            "Duration (s)": duration,
            "Total Steps (steps)": float(total_steps),
            "Cadence (steps/min)": cadence,
            "Walking Speed (m/s)": speed,
            "Mean Step Length (m)": spatial["mean_step_length"],
            "Mean Stride Length (m)": spatial["mean_stride_length"],
            "Mean Stance Time (s)": _safe_mean(temporal["stance_times"]),
            "Mean Swing Time (s)": _safe_mean(temporal["swing_times"]),
            "Gait Cycle Time (s)": cycle_time,
            "Double Limb Support Time (s)": ds_time,
            "Double Support Ratio (%)": ds_ratio,
            "Step Time Coefficient of Variation Right (%)": _safe_cv(temporal["step_times_right"]),
            "Step Time Coefficient of Variation Left (%)": _safe_cv(temporal["step_times_left"]),
            "Stride Time Coefficient of Variation Right (%)": _safe_cv(temporal["stride_times_right"]),
            "Stride Time Coefficient of Variation Left (%)": _safe_cv(temporal["stride_times_left"]),
            "Cadence Variability (%)": _safe_cv(temporal["step_times"]),
            "Step Asymmetry (%)": step_asym,
            "Regularity Index (%)": _regularity_index(temporal["step_times"]),
        }

        diagnostics = {
            "t": sig["t"],
            "ankle_vertical_left": sig["ankle_y_l"],
            "ankle_vertical_right": sig["ankle_y_r"],
            "heel_gap_progression": sig["heel_gap_prog"],
            "step_intervals": np.asarray(temporal["step_times"], dtype=float),
        }
        return AnalysisOutput(metrics=metrics, frame_table=df, events=events, diagnostics=diagnostics)


def print_results_table(metrics: Dict[str, float]) -> None:
    df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    print("\n===== Gait Analysis Results =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_results_csv(metrics: Dict[str, float], out_csv: Path) -> None:
    pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}).to_csv(out_csv, index=False)


def save_diagnostic_plots(out: AnalysisOutput, out_png: Path, fs: float) -> None:
    t = out.diagnostics["t"]
    ay_l = out.diagnostics["ankle_vertical_left"]
    ay_r = out.diagnostics["ankle_vertical_right"]
    heel_gap = out.diagnostics["heel_gap_progression"]
    step_intervals = out.diagnostics["step_intervals"]

    l_hs_t = out.events["left"].heel_strike_idx / fs
    r_hs_t = out.events["right"].heel_strike_idx / fs
    l_to_t = out.events["left"].toe_off_idx / fs
    r_to_t = out.events["right"].toe_off_idx / fs

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    axs[0].plot(t, ay_l, label="Left ankle Y", color="tab:blue")
    axs[0].plot(t, ay_r, label="Right ankle Y", color="tab:orange")
    axs[0].set_title("Ankle Vertical Displacement")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pixels")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    axs[1].plot(t, heel_gap, color="black", linewidth=1.2, label="Heel AP gap (R-L)")
    for x in l_hs_t:
        axs[1].axvline(x, color="tab:blue", linestyle="--", alpha=0.3)
    for x in r_hs_t:
        axs[1].axvline(x, color="tab:orange", linestyle="--", alpha=0.3)
    axs[1].scatter(l_to_t, np.interp(l_to_t, t, heel_gap), color="tab:blue", marker="x", label="L toe-off")
    axs[1].scatter(r_to_t, np.interp(r_to_t, t, heel_gap), color="tab:orange", marker="x", label="R toe-off")
    axs[1].set_title("Detected Gait Events")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Signal")
    axs[1].grid(alpha=0.3)
    axs[1].legend(loc="best")

    axs[2].plot(np.arange(len(step_intervals)), step_intervals, marker="o", color="tab:green")
    if len(step_intervals):
        axs[2].axhline(np.mean(step_intervals), color="red", linestyle="--", label="Mean")
    axs[2].set_title("Step Interval Variability")
    axs[2].set_xlabel("Step index")
    axs[2].set_ylabel("Step interval (s)")
    axs[2].grid(alpha=0.3)
    axs[2].legend(loc="best")

    fig.savefig(out_png, dpi=220)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Markerless gait analysis from monocular video")
    p.add_argument("--video", required=True, help="Path to input walking video")
    p.add_argument("--out_csv", default="gait_metrics.csv", help="Output metrics CSV path")
    p.add_argument("--out_plot", default="gait_diagnostics.png", help="Output diagnostics PNG path")
    p.add_argument("--scale_m_per_px", type=float, default=None, help="Known spatial scale")
    p.add_argument("--subject_height_m", type=float, default=None, help="Subject height for auto-scale")
    p.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--smoothing", choices=["savgol", "butter"], default="savgol")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GaitConfig(frame_skip=max(1, args.frame_skip), smoothing=args.smoothing)
    analyzer = MarkerlessGaitAnalyzer(
        video_path=Path(args.video),
        scale_m_per_px=args.scale_m_per_px,
        subject_height_m=args.subject_height_m,
        config=cfg,
    )

    out = analyzer.analyze()
    print_results_table(out.metrics)
    save_results_csv(out.metrics, Path(args.out_csv))
    save_diagnostic_plots(out, Path(args.out_plot), analyzer.fps / cfg.frame_skip)
    print(f"\nSaved CSV: {args.out_csv}")
    print(f"Saved diagnostic plot: {args.out_plot}")


if __name__ == "__main__":
    main()
