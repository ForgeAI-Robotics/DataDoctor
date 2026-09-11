"""Video integrity checks, including an opt-in full-frame content audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


@dataclass
class VideoAuditOptions:
    """Configuration for the native video check."""

    mode: str = "quick"
    pixel_stride: int = 4
    freeze_mad: float = 0.5
    moving_state_delta: float = 1e-4
    progress: Callable[[int, int], None] | None = field(default=None, repr=False)


@dataclass
class VideoAnomalyInterval:
    """A localized anomaly interval within one physical video file.

    Frame indices are inclusive. Times are relative to the start of the video.
    """

    kind: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass
class VideoAuditResult:
    episode_indices: list[int]
    camera: str
    path: str
    expected_frames: int
    decoded_frames: int = 0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    decode_ok: bool = False
    error: str | None = None
    frame_delta: int | None = None
    pts_nonmonotonic: int = 0
    pts_gap_count: int = 0
    brightness_mean: float | None = None
    contrast_median: float | None = None
    sharpness_p05: float | None = None
    sharpness_median: float | None = None
    black_frames: int = 0
    white_frames: int = 0
    low_contrast_frames: int = 0
    duplicate_transitions: int = 0
    longest_duplicate_run: int = 0
    longest_moving_freeze_run: int = 0
    relative_blur_ratio: float | None = None
    verdict: str = "KEEP"
    reasons: list[str] = field(default_factory=list)
    anomaly_intervals: list[VideoAnomalyInterval] = field(default_factory=list)


def _resolve_video_path(dataset: LoadedDataset, feature: str, episode_meta) -> Path:
    info = dataset.info
    assert info is not None and info.video_path
    episode_index = episode_meta.episode_index
    chunk_index = episode_meta.raw.get(
        f"videos/{feature}/chunk_index", episode_index // (info.chunks_size or 1000)
    )
    file_index = episode_meta.raw.get(
        f"videos/{feature}/file_index",
        episode_meta.raw.get("episode_index", episode_index),
    )
    relative = info.video_path.format(
        video_key=feature,
        episode_chunk=chunk_index,
        episode_index=file_index,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return dataset.root / relative


def _video_groups(dataset: LoadedDataset, feature: str) -> dict[Path, dict]:
    groups: dict[Path, dict] = {}
    for meta in dataset.episodes_meta:
        path = _resolve_video_path(dataset, feature, meta)
        group = groups.setdefault(path, {"episodes": [], "expected_frames": 0})
        group["episodes"].append(meta.episode_index)
        group["expected_frames"] += int(meta.length or 0)
    return groups


def _probe_video(path: Path, count_frames_if_needed: bool = False) -> dict:
    """Return container metadata and verify that the first frame decodes."""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frame_count = int(stream.frames or 0)
            if frame_count == 0 and count_frames_if_needed:
                frame_count = sum(1 for _ in container.decode(video=0))
                container.seek(0)
            can_decode = any(True for _ in container.decode(video=0))
            return {
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "width": int(stream.width or 0),
                "height": int(stream.height or 0),
                "frames": frame_count,
                "can_decode": can_decode,
            }
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("PyAV (av) or OpenCV (cv2) not installed -- skipping video decode checks") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open video")
    try:
        ok, _ = cap.read()
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0) or None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "can_decode": bool(ok),
        }
    finally:
        cap.release()


def _expected_hw(shape: list[int] | None) -> tuple[int, int] | None:
    if not shape or len(shape) < 2:
        return None
    if len(shape) >= 3 and shape[0] in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _true_runs(values: list[bool], min_length: int = 1) -> list[tuple[int, int]]:
    """Return ``(start, end_exclusive)`` for qualifying True runs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                runs.append((start, index))
            start = None
    if start is not None and len(values) - start >= min_length:
        runs.append((start, len(values)))
    return runs


def _frame_time(frame_times: list[float], frame_index: int, fps: float) -> float:
    if 0 <= frame_index < len(frame_times):
        return frame_times[frame_index]
    return frame_index / fps if fps > 0 else 0.0


def _add_transition_intervals(
    result: VideoAuditResult,
    kind: str,
    values: list[bool],
    min_length: int,
    frame_times: list[float],
    fps: float,
) -> None:
    """Localize runs of anomalous adjacent-frame transitions."""
    for start, end_exclusive in _true_runs(values, min_length):
        # Transition j connects frames j and j+1, so a run ending at transition
        # end_exclusive-1 includes frame end_exclusive.
        end_frame = end_exclusive
        start_time = _frame_time(frame_times, start, fps)
        end_time = _frame_time(frame_times, end_frame, fps)
        result.anomaly_intervals.append(VideoAnomalyInterval(
            kind=kind,
            start_frame=start,
            end_frame=end_frame,
            start_time_s=round(start_time, 6),
            end_time_s=round(end_time, 6),
            duration_s=round(max(0.0, end_time - start_time), 6),
        ))


def _add_frame_intervals(
    result: VideoAuditResult,
    kind: str,
    values: list[bool],
    frame_times: list[float],
    fps: float,
) -> None:
    """Localize runs of per-frame anomalies such as black frames."""
    frame_period = 1.0 / fps if fps > 0 else 0.0
    for start, end_exclusive in _true_runs(values):
        end_frame = end_exclusive - 1
        start_time = _frame_time(frame_times, start, fps)
        end_time = _frame_time(frame_times, end_frame, fps)
        result.anomaly_intervals.append(VideoAnomalyInterval(
            kind=kind,
            start_frame=start,
            end_frame=end_frame,
            start_time_s=round(start_time, 6),
            end_time_s=round(end_time, 6),
            duration_s=round(max(frame_period, end_time - start_time + frame_period), 6),
        ))


def _laplacian_variance(gray: np.ndarray) -> float:
    image = gray.astype(np.float32, copy=False)
    if image.shape[0] < 3 or image.shape[1] < 3:
        return 0.0
    center = image[1:-1, 1:-1]
    laplacian = (
        image[:-2, 1:-1] + image[2:, 1:-1]
        + image[1:-1, :-2] + image[1:-1, 2:] - 4.0 * center
    )
    return float(np.var(laplacian))


def _state_motion(dataset: LoadedDataset, episode_indices: list[int]) -> np.ndarray | None:
    by_episode = {episode.episode_index: episode for episode in dataset.episodes_data}
    pieces: list[np.ndarray] = []
    for episode_index in episode_indices:
        episode = by_episode.get(episode_index)
        if episode is None or "observation.state" not in episode.columns:
            return None
        state = np.asarray(episode.columns["observation.state"], dtype=np.float64)
        if len(state) > 1:
            pieces.append(np.linalg.norm(np.diff(state, axis=0), axis=1))
        pieces.append(np.asarray([0.0]))
    return np.concatenate(pieces)[:-1] if pieces else np.empty(0, dtype=np.float64)


def _raise_verdict(result: VideoAuditResult, proposed: str) -> None:
    ranks = {"KEEP": 0, "REVIEW": 1, "DROP_CANDIDATE": 2}
    if ranks[proposed] > ranks[result.verdict]:
        result.verdict = proposed


def _audit_video(
    path: Path,
    episode_indices: list[int],
    camera: str,
    expected_frames: int,
    expected_fps: float,
    expected_shape: list[int] | None,
    state_motion: np.ndarray | None,
    options: VideoAuditOptions,
) -> VideoAuditResult:
    result = VideoAuditResult(episode_indices, camera, str(path), expected_frames)
    if not path.is_file():
        result.error = "video file is missing"
        result.verdict = "DROP_CANDIDATE"
        result.reasons.append(result.error)
        return result

    import av

    brightness: list[float] = []
    contrast: list[float] = []
    sharpness: list[float] = []
    duplicates: list[bool] = []
    moving_freezes: list[bool] = []
    black_or_white: list[bool] = []
    low_contrast: list[bool] = []
    frame_times: list[float] = []
    previous: np.ndarray | None = None
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise RuntimeError("file has no video stream")
            stream = container.streams.video[0]
            result.fps = float(stream.average_rate) if stream.average_rate else None
            result.width = int(stream.width) if stream.width else None
            result.height = int(stream.height) if stream.height else None
            for frame_index, frame in enumerate(container.decode(video=0)):
                gray = frame.to_ndarray(format="gray")
                sample = np.ascontiguousarray(gray[::options.pixel_stride, ::options.pixel_stride])
                mean, std = float(np.mean(sample)), float(np.std(sample))
                brightness.append(mean)
                contrast.append(std)
                sharpness.append(_laplacian_variance(sample))
                is_black = mean <= 5 or float(np.mean(sample <= 8)) >= 0.98
                is_white = mean >= 250 or float(np.mean(sample >= 247)) >= 0.98
                is_low_contrast = std <= 3
                result.black_frames += int(is_black)
                result.white_frames += int(is_white)
                result.low_contrast_frames += int(is_low_contrast)
                black_or_white.append(is_black or is_white)
                low_contrast.append(is_low_contrast)
                frame_times.append(
                    float(frame.time) if frame.time is not None
                    else frame_index / expected_fps if expected_fps > 0 else 0.0
                )
                if previous is not None:
                    mad = float(np.mean(np.abs(sample.astype(np.int16) - previous.astype(np.int16))))
                    duplicate = mad <= options.freeze_mad
                    duplicates.append(duplicate)
                    moving = bool(
                        state_motion is not None and frame_index - 1 < len(state_motion)
                        and state_motion[frame_index - 1] > options.moving_state_delta
                    )
                    moving_freezes.append(duplicate and moving)
                previous = sample
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.decoded_frames = len(brightness)
    result.decode_ok = result.error is None and result.decoded_frames > 0
    result.frame_delta = result.decoded_frames - expected_frames
    if not result.decode_ok:
        result.verdict = "DROP_CANDIDATE"
        result.reasons.append(f"could not fully decode: {result.error or 'no frames'}")
        return result

    result.brightness_mean = float(np.mean(brightness))
    result.contrast_median = float(np.median(contrast))
    result.sharpness_p05 = float(np.percentile(sharpness, 5))
    result.sharpness_median = float(np.median(sharpness))
    result.duplicate_transitions = int(sum(duplicates))
    result.longest_duplicate_run = _longest_true_run(duplicates)
    result.longest_moving_freeze_run = _longest_true_run(moving_freezes)
    pts_nonmonotonic_mask: list[bool] = []
    pts_gap_mask: list[bool] = []
    if len(frame_times) > 1:
        deltas = np.diff(np.asarray(frame_times))
        pts_nonmonotonic_mask = (deltas <= 0).tolist()
        result.pts_nonmonotonic = int(sum(pts_nonmonotonic_mask))
        if expected_fps > 0:
            pts_gap_mask = (deltas > 2.5 / expected_fps).tolist()
            result.pts_gap_count = int(sum(pts_gap_mask))

    if abs(result.frame_delta) > 2:
        _raise_verdict(result, "DROP_CANDIDATE")
        result.reasons.append(f"decoded/metadata frame delta is {result.frame_delta:+d}")
    if result.fps is not None and expected_fps and abs(result.fps - expected_fps) > 1:
        _raise_verdict(result, "DROP_CANDIDATE")
        result.reasons.append(f"fps {result.fps:.3f} != expected {expected_fps:.3f}")
    expected_hw = _expected_hw(expected_shape)
    if expected_hw and (result.height, result.width) != expected_hw:
        _raise_verdict(result, "DROP_CANDIDATE")
        result.reasons.append(
            f"resolution {result.width}x{result.height} != expected {expected_hw[1]}x{expected_hw[0]}"
        )
    if result.pts_nonmonotonic:
        _raise_verdict(result, "DROP_CANDIDATE")
        result.reasons.append(f"{result.pts_nonmonotonic} non-monotonic PTS transitions")
        _add_transition_intervals(
            result, "pts_nonmonotonic", pts_nonmonotonic_mask, 1,
            frame_times, expected_fps,
        )
    if result.pts_gap_count:
        _raise_verdict(result, "REVIEW")
        result.reasons.append(f"{result.pts_gap_count} abnormal PTS gaps")
        _add_transition_intervals(
            result, "pts_gap", pts_gap_mask, 1, frame_times, expected_fps,
        )
    severe = result.black_frames + result.white_frames
    if severe / result.decoded_frames >= 0.1:
        _raise_verdict(result, "DROP_CANDIDATE")
        result.reasons.append(f"black/white frames account for {severe / result.decoded_frames:.1%}")
    elif severe:
        _raise_verdict(result, "REVIEW")
        result.reasons.append(f"{severe} black/white frames")
    if severe:
        _add_frame_intervals(
            result, "black_or_white", black_or_white, frame_times, expected_fps,
        )
    if result.low_contrast_frames / result.decoded_frames >= 0.1:
        _raise_verdict(result, "REVIEW")
        result.reasons.append(
            f"low-contrast frames account for {result.low_contrast_frames / result.decoded_frames:.1%}"
        )
        _add_frame_intervals(
            result, "low_contrast", low_contrast, frame_times, expected_fps,
        )
    if result.longest_moving_freeze_run >= 15:
        _raise_verdict(result, "REVIEW")
        result.reasons.append(
            f"video nearly static for {result.longest_moving_freeze_run} transitions while state moves"
        )
        _add_transition_intervals(
            result, "moving_state_freeze", moving_freezes, 15,
            frame_times, expected_fps,
        )
    elif result.longest_duplicate_run >= 30:
        _raise_verdict(result, "REVIEW")
        result.reasons.append(f"near-duplicate run of {result.longest_duplicate_run} transitions")
        _add_transition_intervals(
            result, "near_duplicate", duplicates, 30, frame_times, expected_fps,
        )
    return result


def _apply_relative_blur(results: list[VideoAuditResult]) -> dict[str, dict[str, float]]:
    by_camera: dict[str, list[float]] = defaultdict(list)
    for item in results:
        if item.decode_ok and item.sharpness_median is not None:
            by_camera[item.camera].append(item.sharpness_median)
    baselines = {
        camera: {"median": float(np.median(values)), "p10": float(np.percentile(values, 10))}
        for camera, values in by_camera.items()
    }
    for item in results:
        if item.sharpness_median is None or item.camera not in baselines:
            continue
        baseline = baselines[item.camera]["median"]
        if baseline <= 0:
            continue
        item.relative_blur_ratio = item.sharpness_median / baseline
        if item.sharpness_median < baselines[item.camera]["p10"] and item.relative_blur_ratio < 0.5:
            _raise_verdict(item, "REVIEW")
            item.reasons.append(f"sharpness is {item.relative_blur_ratio:.1%} of this camera's median")
    return baselines


def _full_video_audit(dataset: LoadedDataset, video_features: dict, options: VideoAuditOptions) -> CheckResult:
    result = CheckResult(name="Video Integrity", severity=Severity.PASS)
    assert dataset.info is not None
    jobs = [
        (camera, spec, path, group)
        for camera, spec in video_features.items()
        for path, group in _video_groups(dataset, camera).items()
    ]
    audited: list[VideoAuditResult] = []
    for completed, (camera, spec, path, group) in enumerate(jobs, start=1):
        audited.append(_audit_video(
            path, group["episodes"], camera, group["expected_frames"],
            float(dataset.info.fps or 0), spec.get("shape"),
            _state_motion(dataset, group["episodes"]), options,
        ))
        if options.progress and (completed % 10 == 0 or completed == len(jobs)):
            options.progress(completed, len(jobs))
    baselines = _apply_relative_blur(audited)
    counts = Counter(item.verdict for item in audited)
    result.details = {
        "mode": "full",
        "settings": {
            "pixel_stride": options.pixel_stride,
            "freeze_mad": options.freeze_mad,
            "moving_state_delta": options.moving_state_delta,
        },
        "video_count": len(audited),
        "expected_frames": sum(item.expected_frames for item in audited),
        "decoded_frames": sum(item.decoded_frames for item in audited),
        "verdict_counts": dict(counts),
        "localized_interval_count": sum(len(item.anomaly_intervals) for item in audited),
        "camera_sharpness_baselines": baselines,
        "videos": [asdict(item) for item in audited],
    }
    result.pass_(f"Fully decoded {len(audited)} video file(s)")
    if counts["DROP_CANDIDATE"]:
        result.fail(f"{counts['DROP_CANDIDATE']} video(s) have structural/decode failures")
    if counts["REVIEW"]:
        result.warn(f"{counts['REVIEW']} video(s) require review for content/timing anomalies")
    if not counts["DROP_CANDIDATE"] and not counts["REVIEW"]:
        result.pass_("No full-audit video anomalies found")
    return result


def check_videos(dataset: LoadedDataset, options: VideoAuditOptions | None = None) -> CheckResult:
    """Check video files in quick mode or run the opt-in full-frame audit."""
    result = CheckResult(name="Video Integrity", severity=Severity.PASS)
    options = options or VideoAuditOptions()
    if dataset.info is None:
        result.fail("Cannot check videos: info.json not loaded")
        return result
    video_features = {
        name: spec for name, spec in dataset.info.features.items() if spec.get("dtype") == "video"
    }
    if not video_features:
        result.pass_("No video features declared -- skipping video checks")
        return result
    if not dataset.info.video_path:
        result.fail("video_path not set in info.json but video features declared")
        return result
    if not dataset.is_local:
        result.warn(f"Found {len(video_features)} video feature(s) -- skipping remote decode checks")
        return result
    if options.mode == "full":
        return _full_video_audit(dataset, video_features, options)

    result.pass_(f"Found {len(video_features)} video feature(s): {list(video_features)}")
    for feature, spec in video_features.items():
        _check_video_feature_quick(dataset, feature, spec, result)
    result.details = {"mode": "quick"}
    return result


def _check_video_feature_quick(dataset: LoadedDataset, feature: str, spec: dict, result: CheckResult) -> None:
    assert dataset.info is not None
    groups = _video_groups(dataset, feature)
    missing = [group["episodes"][0] for path, group in groups.items() if not path.exists()]
    decode_errors: list[int] = []
    fps_mismatches: list[tuple[str, float]] = []
    resolution_mismatches: list[tuple[str, int, int]] = []
    frame_mismatches: list[tuple[str, int, int]] = []
    checked = 0
    for path, group in sorted(groups.items(), key=lambda item: str(item[0])):
        if not path.exists():
            continue
        checked += 1
        try:
            probe = _probe_video(path, count_frames_if_needed=checked <= 20)
            relative = str(path.relative_to(dataset.root))
            if dataset.info.fps and probe["fps"] and abs(probe["fps"] - dataset.info.fps) > 1:
                fps_mismatches.append((relative, probe["fps"]))
            expected_hw = _expected_hw(spec.get("shape"))
            if expected_hw and (probe["height"], probe["width"]) != expected_hw:
                resolution_mismatches.append((relative, probe["height"], probe["width"]))
            if probe["frames"] and group["expected_frames"] and abs(probe["frames"] - group["expected_frames"]) > 2:
                frame_mismatches.append((relative, probe["frames"], group["expected_frames"]))
            if not probe["can_decode"]:
                decode_errors.append(group["episodes"][0])
        except RuntimeError as exc:
            if "not installed" in str(exc):
                result.warn(str(exc))
                return
            decode_errors.append(group["episodes"][0])
        except Exception:
            decode_errors.append(group["episodes"][0])
        if checked >= 20:
            break
    if missing:
        result.fail(f"{feature}: {len(missing)} video file(s) missing (episodes {missing[:5]})")
    else:
        result.pass_(f"{feature}: All video files present")
    if decode_errors:
        result.fail(f"{feature}: {len(decode_errors)} video(s) failed to decode: episodes {decode_errors[:5]}")
    for path, fps in fps_mismatches[:3]:
        result.warn(f"{feature}: {path} video fps={fps:.1f} != dataset fps={dataset.info.fps}")
    for path, height, width in resolution_mismatches[:3]:
        result.warn(f"{feature}: {path} resolution {width}x{height} doesn't match declared shape")
    for path, actual, expected in frame_mismatches[:3]:
        result.warn(f"{feature}: {path} has {actual} frames, expected {expected}")
