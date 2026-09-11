"""Tests for video integrity check."""

import json

import pytest

from lerobot_doctor.checks.videos import VideoAuditOptions, check_videos
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.report import report_to_markdown
from lerobot_doctor.runner import Severity, run_checks
from tests.conftest import create_dataset


def test_no_video_features(tmp_dataset):
    """Dataset without video features should pass."""
    ds = load_local(tmp_dataset)
    result = check_videos(ds)
    assert result.severity == Severity.PASS
    assert any("No video features" in m.message for m in result.messages)


def test_video_feature_declared_but_no_path(tmp_path):
    """Video feature declared but no video_path in info.json."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("video_path" in m.message for m in result.messages)


def test_video_files_missing(tmp_path):
    """Video feature and path declared but files don't exist."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4"
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("missing" in m.message.lower() for m in result.messages)


def test_no_info(tmp_path):
    """Can't check videos without info.json."""
    root = tmp_path / "empty"
    root.mkdir()
    (root / "meta").mkdir()
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL


def test_consolidated_v3_video_shard_passes(tmp_path):
    """Many episodes in one MP4 shard must not trigger per-episode frame-count WARN."""
    from tests.conftest import create_consolidated_v3_dataset

    root = create_consolidated_v3_dataset(tmp_path / "dataset", n_episodes=4, n_frames_per_ep=8, fps=10)
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.PASS
    mismatch_msgs = [
        m.message for m in result.messages
        if m.severity == Severity.WARN and "frames, expected" in m.message
    ]
    assert mismatch_msgs == []


def test_full_video_audit_is_native_and_structured(tmp_path):
    """Full mode decodes every frame and exposes details through CheckResult."""
    from tests.conftest import create_consolidated_v3_dataset

    root = create_consolidated_v3_dataset(
        tmp_path / "dataset", n_episodes=4, n_frames_per_ep=8, fps=10,
    )
    ds = load_local(root)
    result = check_videos(ds, VideoAuditOptions(mode="full", pixel_stride=2))

    assert result.details["mode"] == "full"
    assert result.details["video_count"] == 1
    assert result.details["decoded_frames"] == 32
    assert result.details["expected_frames"] == 32
    video = result.details["videos"][0]
    assert video["episode_indices"] == [0, 1, 2, 3]
    assert video["decode_ok"] is True
    assert not any("resolution" in reason for reason in video["reasons"])
    assert video["anomaly_intervals"]
    interval = video["anomaly_intervals"][0]
    assert interval["kind"] in {"black_or_white", "low_contrast", "moving_state_freeze"}
    assert 0 <= interval["start_frame"] <= interval["end_frame"] < 32
    assert interval["start_time_s"] <= interval["end_time_s"]
    assert interval["duration_s"] > 0

    report = run_checks(
        ds, checks=["videos"],
        video_audit_options=VideoAuditOptions(mode="full", pixel_stride=2),
    )
    markdown = report_to_markdown(report)
    assert "## Full video audit" in markdown
    assert "### Videos requiring attention" in markdown
    assert "### Localized anomaly intervals" in markdown
    assert "Start frame" in markdown
