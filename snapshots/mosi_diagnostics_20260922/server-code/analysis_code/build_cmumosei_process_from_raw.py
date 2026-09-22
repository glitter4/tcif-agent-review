import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile


FACE_SIZE = 112
FACE_MARGIN = 1.45
FACE_DETECTORS = None


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    split: str
    base_id: str
    clip_idx: str
    start: float
    end: float
    raw_segmented_video: str
    output_video: str
    output_audio: str
    output_face: str
    processed_video: str
    processed_audio: str
    processed_face: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a cmumosei-process style dataset root from the raw CMU_MOSEI package."
    )
    parser.add_argument(
        "--raw-root",
        default="/path/to/user/datasets/CMU_MOSEI_extracted_20260408",
        help="Root of the extracted CMU_MOSEI raw package.",
    )
    parser.add_argument(
        "--processed-root",
        default="/path/to/user/datasets/MER-unibench/cmumosei-process",
        help="Existing processed root that already contains label.npz and the ready-made test split.",
    )
    parser.add_argument(
        "--output-root",
        default="/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408",
        help="New dataset root to create. Existing files are reused and never overwritten.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Splits to generate from raw assets. Test is usually linked from the processed root.",
    )
    parser.add_argument(
        "--link-existing-test",
        action="store_true",
        help="Link the ready-made test split from --processed-root into --output-root.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of base-video workers to process in parallel.",
    )
    parser.add_argument(
        "--only-base-ids",
        nargs="*",
        default=None,
        help="Optional subset of base video ids for smoke tests or recovery runs.",
    )
    parser.add_argument(
        "--limit-bases",
        type=int,
        default=None,
        help="Optional cap on the number of base videos to process.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for a JSON summary report. Defaults to <output-root>/build_summary.json.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    tmp = dst.with_name(dst.name + ".tmp-link")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(src, tmp)
    os.replace(tmp, dst)


def ensure_file_link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    try:
        ensure_symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def atomic_save_npy(path: Path, array: np.ndarray):
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_wav(path: Path, sample_rate: int, audio: np.ndarray):
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        wavfile.write(str(tmp_path), sample_rate, audio)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def transcript_map(transcript_path: Path):
    result = {}
    with transcript_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("___", 4)
            if len(parts) < 4:
                continue
            clip_idx = parts[1]
            result[clip_idx] = {
                "start": float(parts[2]),
                "end": float(parts[3]),
                "text": parts[4] if len(parts) > 4 else "",
            }
    return result


def load_label_splits(label_path: Path, splits):
    raw = np.load(label_path, allow_pickle=True)
    key_map = {"train": "train_corpus", "val": "val_corpus", "test": "test_corpus"}
    return {split: raw[key_map[split]].item() for split in splits}


def link_existing_test_split(processed_root: Path, output_root: Path):
    label = load_label_splits(processed_root / "label.npz", ["test"])["test"]
    for subdir, suffix in (("subvideo_new", ".mp4"), ("subaudio", ".wav"), ("openface_face", ".npy")):
        src_dir = processed_root / subdir
        dst_dir = output_root / subdir
        ensure_dir(dst_dir)
        for sample_id in label.keys():
            src = src_dir / f"{sample_id}{suffix}"
            if not src.exists():
                raise FileNotFoundError(f"Processed test asset missing: {src}")
            ensure_symlink(src, dst_dir / f"{sample_id}{suffix}")
    return len(label)


def build_sample_specs(args):
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)

    raw_segmented_video_dir = raw_root / "Raw" / "Videos" / "Segmented" / "Combined"
    transcript_dir = raw_root / "Raw" / "Transcript" / "Segmented" / "Combined"

    labels = load_label_splits(processed_root / "label.npz", args.splits)
    grouped = defaultdict(list)
    transcript_cache = {}

    for split, label_dict in labels.items():
        for sample_id in sorted(label_dict.keys()):
            base_id, clip_idx = sample_id.rsplit("_", 1)
            if args.only_base_ids and base_id not in set(args.only_base_ids):
                continue

            if base_id not in transcript_cache:
                t_path = transcript_dir / f"{base_id}.txt"
                if not t_path.exists():
                    raise FileNotFoundError(f"Transcript file missing for base '{base_id}': {t_path}")
                transcript_cache[base_id] = transcript_map(t_path)

            clip_info = transcript_cache[base_id].get(clip_idx)
            if clip_info is None:
                raise KeyError(f"Clip {sample_id} not found in transcript {base_id}.txt")

            spec = SampleSpec(
                sample_id=sample_id,
                split=split,
                base_id=base_id,
                clip_idx=clip_idx,
                start=clip_info["start"],
                end=clip_info["end"],
                raw_segmented_video=str(raw_segmented_video_dir / f"{sample_id}.mp4"),
                output_video=str(output_root / "subvideo_new" / f"{sample_id}.mp4"),
                output_audio=str(output_root / "subaudio" / f"{sample_id}.wav"),
                output_face=str(output_root / "openface_face" / f"{sample_id}.npy"),
                processed_video=str(processed_root / "subvideo_new" / f"{sample_id}.mp4"),
                processed_audio=str(processed_root / "subaudio" / f"{sample_id}.wav"),
                processed_face=str(processed_root / "openface_face" / f"{sample_id}.npy"),
            )
            grouped[base_id].append(spec)

    base_ids = sorted(grouped.keys())
    if args.limit_bases is not None:
        base_ids = base_ids[: args.limit_bases]
    return [(base_id, sorted(grouped[base_id], key=lambda item: (item.start, item.sample_id))) for base_id in base_ids]


def face_bbox_from_row(row, frame_width, frame_height):
    xs = [float(row[f"x_{idx}"]) for idx in range(68)]
    ys = [float(row[f"y_{idx}"]) for idx in range(68)]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    size = max(max_x - min_x, max_y - min_y) * FACE_MARGIN
    if size <= 1:
        return None
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    left = max(0, int(round(center_x - size * 0.5)))
    top = max(0, int(round(center_y - size * 0.5)))
    right = min(frame_width, int(round(center_x + size * 0.5)))
    bottom = min(frame_height, int(round(center_y + size * 0.5)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def square_bbox_from_rect(x, y, w, h, frame_width, frame_height, margin=1.35):
    size = max(w, h) * margin
    center_x = x + w * 0.5
    center_y = y + h * 0.5
    left = max(0, int(round(center_x - size * 0.5)))
    top = max(0, int(round(center_y - size * 0.5)))
    right = min(frame_width, int(round(center_x + size * 0.5)))
    bottom = min(frame_height, int(round(center_y + size * 0.5)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def largest_face_bbox(gray_frame, frame_width, frame_height):
    global FACE_DETECTORS
    if FACE_DETECTORS is None:
        cascade_paths = [
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
        ]
        FACE_DETECTORS = [cv2.CascadeClassifier(path) for path in cascade_paths]
    for detector in FACE_DETECTORS:
        faces = detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(36, 36),
        )
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        bbox = square_bbox_from_rect(x, y, w, h, frame_width, frame_height)
        if bbox is not None:
            return bbox
    return None


def build_face_track_with_detector(full_video_path: Path, sample: SampleSpec, fps: float):
    cap = cv2.VideoCapture(str(full_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open full video for detector fallback: {full_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(math.floor(sample.start * fps)))
    end_frame = max(start_frame, int(math.ceil(sample.end * fps)) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    face_frames = []

    try:
        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = largest_face_bbox(gray, frame_width, frame_height)
            if bbox is not None:
                left, top, right, bottom = bbox
                crop = frame[top:bottom, left:right]
                if crop.size > 0:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
                    face_frames.append(crop)
            current_frame += 1
    finally:
        cap.release()

    if face_frames:
        return np.stack(face_frames, axis=0).astype(np.uint8), False
    return build_empty_face_track(sample, fps), True


def build_empty_face_track(sample: SampleSpec, fps: float):
    start_frame = max(0, int(math.floor(sample.start * fps)))
    end_frame = max(start_frame, int(math.ceil(sample.end * fps)) - 1)
    length = max(1, end_frame - start_frame + 1)
    return np.zeros((length, FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)


def parse_openface_rows(csv_path: Path, samples, fps, frame_width, frame_height):
    needs_face = [sample for sample in samples if not Path(sample.output_face).exists()]
    if not needs_face:
        return {}

    rows_by_frame = defaultdict(list)
    pointer = 0
    needs_face = sorted(needs_face, key=lambda item: (item.start, item.end))

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            timestamp = float(row["timestamp"])
            while pointer < len(needs_face) and timestamp > needs_face[pointer].end:
                pointer += 1
            if pointer >= len(needs_face):
                break
            sample = needs_face[pointer]
            if timestamp < sample.start or timestamp > sample.end:
                continue
            if float(row["success"]) < 0.5 or float(row["confidence"]) < 0.5:
                continue
            bbox = face_bbox_from_row(row, frame_width, frame_height)
            if bbox is None:
                continue
            frame_idx = int(row["frame"]) - 1
            if frame_idx < 0:
                frame_idx = max(0, int(round(timestamp * fps)))
            rows_by_frame[frame_idx].append((sample.sample_id, bbox))

    return rows_by_frame


def audio_slice(full_audio_path: Path, samples):
    sample_rate, audio = wavfile.read(str(full_audio_path))
    for sample in samples:
        out_path = Path(sample.output_audio)
        if out_path.exists():
            continue
        start_idx = max(0, int(math.floor(sample.start * sample_rate)))
        end_idx = min(audio.shape[0], int(math.ceil(sample.end * sample_rate)))
        if end_idx <= start_idx:
            raise RuntimeError(f"Audio slice is empty for sample {sample.sample_id}")
        atomic_write_wav(out_path, sample_rate, np.ascontiguousarray(audio[start_idx:end_idx]))


def create_video_writer(path: Path, fps, frame_width, frame_height):
    ensure_dir(path.parent)
    tmp_path = path.with_name(path.name + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    return writer, tmp_path


def finalize_video_writer(writer, tmp_path: Path, final_path: Path):
    writer.release()
    os.replace(tmp_path, final_path)


def process_base(job):
    base_id, samples, raw_root_str = job
    raw_root = Path(raw_root_str)

    full_video_path = raw_root / "Raw" / "Videos" / "Full" / "Combined" / f"{base_id}.mp4"
    full_audio_path = raw_root / "Raw" / "Audio" / "Full" / "WAV_16000" / f"{base_id}.wav"
    openface_csv_path = raw_root / "Raw" / "Videos" / "Full" / "OpenFace2.0" / f"{base_id}.csv"

    if not full_video_path.exists():
        raise FileNotFoundError(f"Full video missing for base '{base_id}': {full_video_path}")
    if not full_audio_path.exists():
        raise FileNotFoundError(f"Full audio missing for base '{base_id}': {full_audio_path}")
    if not openface_csv_path.exists():
        raise FileNotFoundError(f"OpenFace CSV missing for base '{base_id}': {openface_csv_path}")

    linked_videos = 0
    generated_videos = 0
    generated_audios = 0
    generated_faces = 0
    fallback_faces = 0
    black_faces = 0

    # Reuse any raw segmented video clips that already exist.
    missing_video_samples = []
    for sample in samples:
        out_video = Path(sample.output_video)
        if out_video.exists():
            continue
        raw_seg = Path(sample.raw_segmented_video)
        if raw_seg.exists():
            ensure_symlink(raw_seg, out_video)
            linked_videos += 1
        else:
            missing_video_samples.append(sample)

    audio_missing = [sample for sample in samples if not Path(sample.output_audio).exists()]
    if audio_missing:
        audio_slice(full_audio_path, audio_missing)
        generated_audios += len(audio_missing)

    face_missing = [sample for sample in samples if not Path(sample.output_face).exists()]
    if not missing_video_samples and not face_missing:
        return {
            "base_id": base_id,
            "linked_videos": linked_videos,
            "generated_videos": generated_videos,
            "generated_audios": generated_audios,
            "generated_faces": generated_faces,
            "fallback_faces": fallback_faces,
            "black_faces": black_faces,
        }

    cap = cv2.VideoCapture(str(full_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open full video: {full_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for {full_video_path}")

    rows_by_frame = parse_openface_rows(openface_csv_path, samples, fps, frame_width, frame_height)
    face_buffers = defaultdict(list)

    video_specs = []
    for sample in missing_video_samples:
        start_frame = max(0, int(math.floor(sample.start * fps)))
        end_frame = max(start_frame, int(math.ceil(sample.end * fps)) - 1)
        video_specs.append((start_frame, end_frame, sample))
    video_specs.sort(key=lambda item: item[0])

    needed_frames = []
    if rows_by_frame:
        needed_frames.extend(rows_by_frame.keys())
    if video_specs:
        needed_frames.extend([video_specs[0][0], video_specs[-1][1]])
    if not needed_frames:
        for sample in face_missing:
            out_path = Path(sample.output_face)
            if out_path.exists():
                continue
            fallback_track, used_black_track = build_face_track_with_detector(full_video_path, sample, fps)
            atomic_save_npy(out_path, fallback_track)
            generated_faces += 1
            fallback_faces += 1
            black_faces += int(used_black_track)
        cap.release()
        return {
            "base_id": base_id,
            "linked_videos": linked_videos,
            "generated_videos": generated_videos,
            "generated_audios": generated_audios,
            "generated_faces": generated_faces,
            "fallback_faces": fallback_faces,
            "black_faces": black_faces,
        }

    start_seek = max(0, min(needed_frames))
    end_seek = max(needed_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_seek)
    current_frame = start_seek

    active_writer = None
    active_tmp = None
    active_spec = None
    video_index = 0

    try:
        while current_frame <= end_seek:
            ok, frame = cap.read()
            if not ok:
                break

            while video_index < len(video_specs) and current_frame > video_specs[video_index][1]:
                video_index += 1

            if active_writer is None and video_index < len(video_specs):
                start_frame, end_frame, sample = video_specs[video_index]
                if start_frame <= current_frame <= end_frame:
                    active_writer, active_tmp = create_video_writer(
                        Path(sample.output_video), fps, frame_width, frame_height
                    )
                    active_spec = (start_frame, end_frame, sample)

            if active_writer is not None:
                _, end_frame, sample = active_spec
                if current_frame <= end_frame:
                    active_writer.write(frame)
                if current_frame >= end_frame:
                    finalize_video_writer(active_writer, active_tmp, Path(sample.output_video))
                    generated_videos += 1
                    active_writer = None
                    active_tmp = None
                    active_spec = None
                    video_index += 1

            frame_rows = rows_by_frame.get(current_frame)
            if frame_rows:
                for sample_id, bbox in frame_rows:
                    left, top, right, bottom = bbox
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
                    face_buffers[sample_id].append(crop)

            current_frame += 1
    finally:
        if active_writer is not None:
            _, _, sample = active_spec
            finalize_video_writer(active_writer, active_tmp, Path(sample.output_video))
            generated_videos += 1
        cap.release()

    for sample in face_missing:
        out_path = Path(sample.output_face)
        if out_path.exists():
            continue
        frames = face_buffers.get(sample.sample_id)
        if not frames:
            fallback_track, used_black_track = build_face_track_with_detector(full_video_path, sample, fps)
            atomic_save_npy(out_path, fallback_track)
            generated_faces += 1
            fallback_faces += 1
            black_faces += int(used_black_track)
            continue
        atomic_save_npy(out_path, np.stack(frames, axis=0).astype(np.uint8))
        generated_faces += 1

    return {
        "base_id": base_id,
        "linked_videos": linked_videos,
        "generated_videos": generated_videos,
        "generated_audios": generated_audios,
        "generated_faces": generated_faces,
        "fallback_faces": fallback_faces,
        "black_faces": black_faces,
    }


def prepare_output_root(processed_root: Path, output_root: Path):
    ensure_dir(output_root)
    ensure_dir(output_root / "subvideo_new")
    ensure_dir(output_root / "subaudio")
    ensure_dir(output_root / "openface_face")
    ensure_file_link_or_copy(processed_root / "label.npz", output_root / "label.npz")
    transcription = processed_root / "transcription-engchi-polish.csv"
    if transcription.exists():
        ensure_file_link_or_copy(transcription, output_root / "transcription-engchi-polish.csv")


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)

    prepare_output_root(processed_root, output_root)

    linked_test = 0
    if args.link_existing_test:
        linked_test = link_existing_test_split(processed_root, output_root)
        print(f"Linked existing test split: {linked_test} samples")

    jobs = build_sample_specs(args)
    print(f"Prepared {len(jobs)} base jobs for splits={args.splits}")

    summary = {
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "linked_test_samples": linked_test,
        "base_jobs": len(jobs),
        "results": [],
    }

    totals = {
        "linked_videos": 0,
        "generated_videos": 0,
        "generated_audios": 0,
        "generated_faces": 0,
        "fallback_faces": 0,
        "black_faces": 0,
    }

    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = {
            pool.submit(process_base, (base_id, samples, str(raw_root))): base_id
            for base_id, samples in jobs
        }
        completed = 0
        for future in as_completed(futures):
            base_id = futures[future]
            result = future.result()
            summary["results"].append(result)
            for key in totals:
                totals[key] += result[key]
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] base={base_id} "
                    f"linked_videos={totals['linked_videos']} "
                    f"generated_videos={totals['generated_videos']} "
                    f"generated_audios={totals['generated_audios']} "
                    f"generated_faces={totals['generated_faces']} "
                    f"fallback_faces={totals['fallback_faces']} "
                    f"black_faces={totals['black_faces']}"
                )

    summary["totals"] = totals
    summary_path = Path(args.summary_json) if args.summary_json else output_root / "build_summary.json"
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"linked_test_samples": linked_test, **totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
