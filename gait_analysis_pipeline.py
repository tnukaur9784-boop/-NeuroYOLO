#!/usr/bin/env python3
"""Robust markerless monocular gait analysis pipeline using MediaPipe Pose.

Usage:
    python gait_analysis_pipeline.py --video path/to/video.mp4 --output_dir outputs

Notes
-----
- Designed for single-subject walking videos (side or frontal view).
- Computes temporal/spatial gait metrics with explicit left/right event handling.
- Uses adaptive event detection and signal-quality fallbacks for irregular gait.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


# MediaPipe landmark indices
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
L_SHOULDER, R_SHOULDER = 11, 12


@dataclass
class GaitConfig:
    frame_skip: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_window_sec: float = 0.25
    butter_cutoff_hz: float = 5.0
    use_savgol: bool = True
    subject_height_m: float = 1.70
    min_step_time_s: float = 0.22
    max_step_time_s: float = 1.80


class MarkerlessGaitAnalyzer:
    """End-to-end gait pipeline with robust event detection and variability metrics."""

    def __init__(self, video_path: str, config: Optional[GaitConfig] = None):
        self.video_path = Path(video_path)
        self.config = config or GaitConfig()

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.fps: float = 0.0
        self.n_frames_raw: int = 0
        self.df: Optional[pd.DataFrame] = None
        self.scale_m_per_px: Optional[float] = None
        self.events: Dict[str, np.ndarray] = {}

    @staticmethod
    def _interp_nan(signal: np.ndarray) -> np.ndarray:
        s = pd.Series(signal)
        return s.interpolate(limit_direction="both").bfill().ffill().to_numpy()

    def _smooth(self, signal: np.ndarray, fps_eff: float) -> np.ndarray:
        x = self._interp_nan(signal)
        if len(x) < 5:
            return x

        if self.config.use_savgol:
            window = max(5, int(self.config.smooth_window_sec * fps_eff) | 1)
            if window >= len(x):
                window = (len(x) - 1) if len(x) % 2 == 0 else len(x)
                window = max(5, window)
                if window % 2 == 0:
                    window -= 1
            try:
                x = savgol_filter(x, window_length=window, polyorder=2, mode="interp")
            except ValueError:
                pass

        nyq = 0.5 * fps_eff
        cutoff = min(self.config.butter_cutoff_hz, nyq * 0.95)
        if cutoff > 0.2 and len(x) > 12:
            b, a = butter(2, cutoff / nyq, btype="low")
            x = filtfilt(b, a, x)

        return x

    def extract_pose_timeseries(self) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        recs: List[Dict[str, float]] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % self.config.frame_skip != 0:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            row = {"frame": idx, "t": idx / self.fps}
            for i, p in enumerate(lm):
                row[f"x_{i}"] = p.x * w
                row[f"y_{i}"] = p.y * h
                row[f"v_{i}"] = p.visibility
            recs.append(row)

        cap.release()
        pose.close()

        if not recs:
            raise RuntimeError("No valid pose landmarks detected.")

        df = pd.DataFrame.from_records(recs).reset_index(drop=True)

        # Normalize coordinates to reduce camera/depth scaling effects.
        df["midhip_x"] = (df[f"x_{L_HIP}"] + df[f"x_{R_HIP}"]) / 2
        df["midhip_y"] = (df[f"y_{L_HIP}"] + df[f"y_{R_HIP}"]) / 2
        df["shoulder_mid_y"] = (df[f"y_{L_SHOULDER}"] + df[f"y_{R_SHOULDER}"]) / 2
        df["body_px"] = (df[[f"y_{L_ANKLE}", f"y_{R_ANKLE}"]].mean(axis=1) - df["shoulder_mid_y"]).abs()
        body_px = self._interp_nan(df["body_px"].to_numpy())
        body_px = np.clip(body_px, np.percentile(body_px, 5), np.percentile(body_px, 95))
        df["body_px"] = body_px

        self.scale_m_per_px = self.config.subject_height_m / float(np.nanmedian(body_px))

        # Relative AP trajectories wrt pelvis for robustness when camera follows subject.
        for side, ankle in [("L", L_ANKLE), ("R", R_ANKLE)]:
            df[f"{side}_ankle_ap"] = (df[f"x_{ankle}"] - df["midhip_x"]) / df["body_px"]

        # Vertical trajectories normalized by body size (y increases downward in image).
        for side, heel, toe, ankle in [("L", L_HEEL, L_TOE, L_ANKLE), ("R", R_HEEL, R_TOE, R_ANKLE)]:
            df[f"{side}_heel_y_n"] = df[f"y_{heel}"] / df["body_px"]
            df[f"{side}_toe_y_n"] = df[f"y_{toe}"] / df["body_px"]
            df[f"{side}_ankle_y_n"] = df[f"y_{ankle}"] / df["body_px"]

        self.df = df
        return df

    def _detect_events_side(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Detect heel-strike (HS) and toe-off (TO) for one side.

        Primary logic (side-view robust):
        - HS near maximal forward AP position of foot relative to pelvis.
        - TO near maximal backward AP position.

        Fallback logic (frontal/noisy):
        - HS from heel-down peaks in normalized heel-y.
        - TO from toe-lift minima in normalized toe-y.
        """
        assert self.df is not None
        fps_eff = self.fps / self.config.frame_skip
        min_dist = max(2, int(self.config.min_step_time_s * fps_eff * 0.8))

        ap = self._smooth(self.df[f"{side}_ankle_ap"].to_numpy(), fps_eff)
        heel_y = self._smooth(self.df[f"{side}_heel_y_n"].to_numpy(), fps_eff)
        toe_y = self._smooth(self.df[f"{side}_toe_y_n"].to_numpy(), fps_eff)

        # Determine dominant progression direction from pelvis path projected on x.
        prog_sign = 1.0 if np.nanmedian(np.diff(self.df["midhip_x"])) >= 0 else -1.0
        ap_signed = ap * prog_sign

        hs_ap, _ = find_peaks(ap_signed, distance=min_dist, prominence=max(np.std(ap_signed) * 0.15, 0.02))
        to_ap, _ = find_peaks(-ap_signed, distance=min_dist, prominence=max(np.std(ap_signed) * 0.15, 0.02))

        # Fallback if too few AP events.
        if len(hs_ap) < 2:
            hs_ap, _ = find_peaks(heel_y, distance=min_dist, prominence=max(np.std(heel_y) * 0.2, 0.01))
        if len(to_ap) < 2:
            to_ap, _ = find_peaks(-toe_y, distance=min_dist, prominence=max(np.std(toe_y) * 0.2, 0.01))

        hs = np.array(sorted(hs_ap), dtype=int)
        to = np.array(sorted(to_ap), dtype=int)

        # Remove physiologically implausible intervals.
        def clean_series(events: np.ndarray) -> np.ndarray:
            if len(events) < 2:
                return events
            keep = [events[0]]
            for e in events[1:]:
                dt = (e - keep[-1]) / fps_eff
                if self.config.min_step_time_s <= dt <= self.config.max_step_time_s:
                    keep.append(e)
            return np.array(keep, dtype=int)

        return clean_series(hs), clean_series(to)

    def detect_gait_events(self) -> Dict[str, np.ndarray]:
        hs_l, to_l = self._detect_events_side("L")
        hs_r, to_r = self._detect_events_side("R")
        self.events = {"HS_L": hs_l, "TO_L": to_l, "HS_R": hs_r, "TO_R": to_r}
        return self.events

    def _build_stance_swing(self, hs: np.ndarray, to: np.ndarray) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
        fps_eff = self.fps / self.config.frame_skip
        stance, swing = [], []
        stance_intervals: List[Tuple[float, float]] = []
        if len(hs) < 2:
            return stance, swing, stance_intervals

        for i in range(len(hs) - 1):
            h0, h1 = hs[i], hs[i + 1]
            to_between = to[(to > h0) & (to < h1)]
            if len(to_between) == 0:
                # fallback split if TO is missed in irregular gait
                tc = h0 + int(0.62 * (h1 - h0))
            else:
                tc = int(to_between[0])

            st = (tc - h0) / fps_eff
            sw = (h1 - tc) / fps_eff
            cyc = (h1 - h0) / fps_eff
            if 0.05 < st < cyc and 0.05 < sw < cyc:
                stance.append(st)
                swing.append(sw)
                stance_intervals.append((h0 / fps_eff, tc / fps_eff))
        return stance, swing, stance_intervals

    @staticmethod
    def _cv(x: Sequence[float]) -> float:
        arr = np.array(x, dtype=float)
        if len(arr) < 2 or np.mean(arr) <= 1e-8:
            return 0.0
        return float(np.std(arr, ddof=1) / np.mean(arr) * 100.0)

    def compute_metrics(self) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
        if self.df is None:
            self.extract_pose_timeseries()
        if not self.events:
            self.detect_gait_events()

        df = self.df
        assert df is not None and self.scale_m_per_px is not None

        fps_eff = self.fps / self.config.frame_skip
        duration_s = (len(df) - 1) / fps_eff

        hs_l, hs_r = self.events["HS_L"], self.events["HS_R"]
        to_l, to_r = self.events["TO_L"], self.events["TO_R"]

        # Step times from alternating bilateral HS sequence.
        merged = sorted([(int(i), "L") for i in hs_l] + [(int(i), "R") for i in hs_r], key=lambda x: x[0])
        step_times_all, step_times_l, step_times_r = [], [], []
        for (f0, s0), (f1, s1) in zip(merged[:-1], merged[1:]):
            if s0 == s1:
                continue
            dt = (f1 - f0) / fps_eff
            if self.config.min_step_time_s <= dt <= self.config.max_step_time_s:
                step_times_all.append(dt)
                if s1 == "L":
                    step_times_l.append(dt)
                else:
                    step_times_r.append(dt)

        stride_l = np.diff(hs_l) / fps_eff if len(hs_l) > 1 else np.array([])
        stride_r = np.diff(hs_r) / fps_eff if len(hs_r) > 1 else np.array([])

        # Speed from pelvis displacement along principal progression axis.
        pelvis_xy = np.column_stack([df["midhip_x"].to_numpy(), df["midhip_y"].to_numpy()])
        pelvis_xy = pelvis_xy - np.nanmean(pelvis_xy, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(np.nan_to_num(pelvis_xy), full_matrices=False)
        axis = vt[0]
        proj = pelvis_xy @ axis
        distance_m = abs(proj[-1] - proj[0]) * self.scale_m_per_px
        speed_mps = distance_m / duration_s if duration_s > 0 else 0.0

        # Spatial steps from speed * temporal steps (robust when depth scaling fluctuates).
        step_lengths = [speed_mps * s for s in step_times_all]
        stride_lengths_l = [speed_mps * s for s in stride_l]
        stride_lengths_r = [speed_mps * s for s in stride_r]

        stance_l, swing_l, stance_int_l = self._build_stance_swing(hs_l, to_l)
        stance_r, swing_r, stance_int_r = self._build_stance_swing(hs_r, to_r)

        # Double support: overlap of left/right stance intervals.
        ds_segments = []
        i = j = 0
        while i < len(stance_int_l) and j < len(stance_int_r):
            a0, a1 = stance_int_l[i]
            b0, b1 = stance_int_r[j]
            start, end = max(a0, b0), min(a1, b1)
            if end > start:
                ds_segments.append(end - start)
            if a1 < b1:
                i += 1
            else:
                j += 1

        mean_cycle = float(np.mean(np.r_[stride_l, stride_r])) if len(np.r_[stride_l, stride_r]) else 0.0
        mean_ds = float(np.mean(ds_segments)) if ds_segments else 0.0
        ds_ratio = (mean_ds / mean_cycle * 100.0) if mean_cycle > 0 else 0.0

        total_steps = len(step_times_all)
        cadence = (total_steps / duration_s) * 60.0 if duration_s > 0 else 0.0

        # Variability + asymmetry metrics.
        cadence_var = self._cv(step_times_all)
        step_cv_r = self._cv(step_times_r)
        step_cv_l = self._cv(step_times_l)
        stride_cv_r = self._cv(stride_r)
        stride_cv_l = self._cv(stride_l)

        mean_step_r = np.mean([speed_mps * s for s in step_times_r]) if step_times_r else 0.0
        mean_step_l = np.mean([speed_mps * s for s in step_times_l]) if step_times_l else 0.0
        step_asym = (abs(mean_step_r - mean_step_l) / ((mean_step_r + mean_step_l) / 2.0) * 100.0) if (mean_step_r + mean_step_l) > 1e-8 else 0.0

        # Regularity index from autocorrelation of bilateral ankle vertical signal.
        b_sig = self._smooth(((df["L_ankle_y_n"] if "L_ankle_y_n" in df else df[f"L_ankle_y_n"]) + (df["R_ankle_y_n"] if "R_ankle_y_n" in df else df[f"R_ankle_y_n"])) / 2.0, fps_eff)
        b_sig = np.asarray(b_sig) - np.mean(b_sig)
        ac = np.correlate(b_sig, b_sig, mode="full")
        ac = ac[len(ac) // 2 :]
        ac = ac / (ac[0] + 1e-12)
        lag_step = int(np.mean(step_times_all) * fps_eff) if step_times_all else 0
        lag_stride = int(np.mean(np.r_[stride_l, stride_r]) * fps_eff) if len(np.r_[stride_l, stride_r]) else 0
        reg_step = ac[lag_step] if 0 < lag_step < len(ac) else 0.0
        reg_stride = ac[lag_stride] if 0 < lag_stride < len(ac) else 0.0
        regularity_index = float(np.clip((abs(reg_step) + abs(reg_stride)) / 2.0 * 100.0, 0, 100))

        results = {
            "Duration (s)": duration_s,
            "Total Steps (steps)": float(total_steps),
            "Cadence (steps/min)": cadence,
            "Walking Speed (m/s)": speed_mps,
            "Mean Step Length (m)": float(np.mean(step_lengths)) if step_lengths else 0.0,
            "Mean Stride Length (m)": float(np.mean(stride_lengths_l + stride_lengths_r)) if (stride_lengths_l or stride_lengths_r) else 0.0,
            "Mean Stance Time (s)": float(np.mean(stance_l + stance_r)) if (stance_l or stance_r) else 0.0,
            "Mean Swing Time (s)": float(np.mean(swing_l + swing_r)) if (swing_l or swing_r) else 0.0,
            "Gait Cycle Time (s)": mean_cycle,
            "Double Limb Support Time (s)": mean_ds,
            "Double Support Ratio (%)": ds_ratio,
            "Step Time CV Right (%)": step_cv_r,
            "Step Time CV Left (%)": step_cv_l,
            "Stride Time CV Right (%)": stride_cv_r,
            "Stride Time CV Left (%)": stride_cv_l,
            "Cadence Variability (%)": cadence_var,
            "Step Asymmetry (%)": step_asym,
            "Regularity Index (%)": regularity_index,
        }

        diagnostics = {
            "t": df["t"].to_numpy(),
            "L_ankle_y_n": self._smooth(df["L_ankle_y_n"].to_numpy(), fps_eff),
            "R_ankle_y_n": self._smooth(df["R_ankle_y_n"].to_numpy(), fps_eff),
            "step_times": np.array(step_times_all, dtype=float),
            "HS_L_t": hs_l / fps_eff,
            "HS_R_t": hs_r / fps_eff,
            "TO_L_t": to_l / fps_eff,
            "TO_R_t": to_r / fps_eff,
        }

        out_df = pd.DataFrame({"Metric": list(results.keys()), "Value": list(results.values())})
        return out_df, results, diagnostics


def save_diagnostic_plots(diag: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)

    # 1) Ankle vertical displacement + events
    ax = axes[0]
    ax.plot(diag["t"], diag["L_ankle_y_n"], label="Left ankle vertical (norm)", lw=1.8)
    ax.plot(diag["t"], diag["R_ankle_y_n"], label="Right ankle vertical (norm)", lw=1.8)
    for t in diag["HS_L_t"]:
        ax.axvline(t, color="tab:blue", alpha=0.18)
    for t in diag["HS_R_t"]:
        ax.axvline(t, color="tab:orange", alpha=0.18)
    ax.set_title("Ankle Vertical Displacement + Heel-Strike Events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized y")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # 2) Event raster plot
    ax = axes[1]
    for y, key, color in [
        (3, "HS_L_t", "tab:blue"),
        (2, "TO_L_t", "tab:cyan"),
        (1, "HS_R_t", "tab:orange"),
        (0, "TO_R_t", "tab:red"),
    ]:
        times = diag[key]
        ax.scatter(times, np.full_like(times, y, dtype=float), s=36, c=color, label=key)
    ax.set_yticks([0, 1, 2, 3], ["TO_R", "HS_R", "TO_L", "HS_L"])
    ax.set_title("Detected Gait Events")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2)

    # 3) Step interval variability
    ax = axes[2]
    step_t = diag["step_times"]
    if len(step_t):
        ax.plot(np.arange(1, len(step_t) + 1), step_t, marker="o", lw=1.5)
        mean_s = np.mean(step_t)
        sd_s = np.std(step_t, ddof=1) if len(step_t) > 1 else 0.0
        ax.axhline(mean_s, color="k", ls="--", label="Mean")
        ax.axhline(mean_s + sd_s, color="gray", ls=":", label="±1 SD")
        ax.axhline(mean_s - sd_s, color="gray", ls=":")
    ax.set_title("Step Interval Variability")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Step time (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "gait_diagnostic_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_results_table(result_df: pd.DataFrame) -> None:
    printable = result_df.copy()
    printable["Value"] = printable["Value"].map(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
    print("\n=== Gait Analysis Results ===")
    print(printable.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Markerless monocular gait analysis from walking video.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_dir", type=str, default="gait_outputs", help="Directory for CSV and plots")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--subject_height_m", type=float, default=1.70, help="Subject height in meters for pixel->meter scaling")
    args = parser.parse_args()

    cfg = GaitConfig(frame_skip=max(1, args.frame_skip), subject_height_m=args.subject_height_m)
    analyzer = MarkerlessGaitAnalyzer(args.video, cfg)

    result_df, _, diag = analyzer.compute_metrics()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gait_metrics.csv"
    result_df.to_csv(csv_path, index=False)

    save_diagnostic_plots(diag, out_dir)
    print_results_table(result_df)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved diagnostics: {out_dir / 'gait_diagnostic_plots.png'}")


if __name__ == "__main__":
    main()
