import os
import csv
import json
import math
import random
import re
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import wavfile
import transformers.utils
import transformers.utils.import_utils as _import_utils

def _disable_sklearn():
    def _false():
        return False
    transformers.utils.is_sklearn_available = _false
    _import_utils.is_sklearn_available = _false

_disable_sklearn()

from transformers import AutoTokenizer
from torchvision.io import read_video

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None
    _HAS_TORCHAUDIO = False


class MultimodalEmbeddingCache:
    def __init__(self, root, split):
        self.root = os.path.abspath(os.path.expanduser(str(root))) if root else ""
        self.split = split
        self.split_dir = os.path.join(self.root, split) if self.root else ""
        self.index = {}
        self.shard_dirs = []
        self._arrays = {}

        if not self.root:
            return
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Embedding cache split directory not found: {self.split_dir}")

        for name in sorted(os.listdir(self.split_dir)):
            shard_dir = os.path.join(self.split_dir, name)
            if not os.path.isdir(shard_dir):
                continue
            ids_path = os.path.join(shard_dir, "ids.json")
            done_path = os.path.join(shard_dir, "done.json")
            if not os.path.isfile(ids_path) or not os.path.isfile(done_path):
                continue
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            shard_idx = len(self.shard_dirs)
            self.shard_dirs.append(shard_dir)
            for row_idx, sample_id in enumerate(ids):
                self.index[str(sample_id)] = (shard_idx, row_idx)

        if not self.index:
            raise RuntimeError(f"No completed embedding cache shards found under {self.split_dir}")

    def __len__(self):
        return len(self.index)

    def __contains__(self, sample_id):
        return str(sample_id) in self.index

    def _array(self, shard_idx, name):
        key = (shard_idx, name)
        if key not in self._arrays:
            path = os.path.join(self.shard_dirs[shard_idx], f"{name}.npy")
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def get(self, sample_id):
        loc = self.index.get(str(sample_id))
        if loc is None:
            return None
        shard_idx, row_idx = loc

        def tensor(name):
            arr = self._array(shard_idx, name)[row_idx]
            return torch.from_numpy(np.array(arr, copy=True))

        return {
            "cached_vision": tensor("vision"),
            "cached_text": tensor("text"),
            "cached_audio": tensor("audio"),
            "cached_text_mask": tensor("text_mask").to(torch.long),
            "cached_audio_mask": tensor("audio_mask").to(torch.long),
        }

def _validate_embedding_cache(cache, ids, split):
    if cache is None:
        return
    missing = [sample_id for sample_id in ids if sample_id not in cache]
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise RuntimeError(
            f"Embedding cache for split '{split}' is incomplete: missing {len(missing)} samples. "
            f"First missing ids: {preview}"
        )
    print(f"Loaded embedding cache split={split} samples={len(ids)} root={cache.root}")

def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform is None:
        raise ValueError("Waveform is None")
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() == 2:
        if waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
        return waveform.mean(dim=1)
    return waveform.reshape(-1)

def _resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if int(orig_sr) == int(target_sr):
        return waveform
    if waveform.numel() == 0:
        return waveform
    if _HAS_TORCHAUDIO:
        return torchaudio.functional.resample(waveform.unsqueeze(0), orig_sr, target_sr).squeeze(0)
    new_len = max(1, int(round(waveform.numel() * float(target_sr) / float(orig_sr))))
    resampled = F.interpolate(waveform.view(1, 1, -1), size=new_len, mode="linear", align_corners=False)
    return resampled.view(-1)

def _spectral_denoise(waveform: torch.Tensor, n_fft: int, hop_length: int, noise_frames: int) -> torch.Tensor:
    if waveform.numel() < n_fft:
        return waveform
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    mag = stft.abs()
    if mag.numel() == 0:
        return waveform
    nf = max(1, min(noise_frames, mag.shape[1]))
    noise_profile = mag[:, :nf].median(dim=1).values
    thresh = noise_profile.unsqueeze(1) * 1.2
    mask = (mag >= thresh).to(stft.dtype)
    stft_clean = stft * mask
    return torch.istft(stft_clean, n_fft=n_fft, hop_length=hop_length, length=waveform.numel())

def _frame_waveform(waveform: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    if frame_len <= 0 or hop_len <= 0:
        return waveform.unsqueeze(0)
    if waveform.numel() < frame_len:
        waveform = F.pad(waveform, (0, frame_len - waveform.numel()))
    frames = waveform.unfold(0, frame_len, hop_len)
    if frames.numel() == 0:
        frames = waveform[:frame_len].unsqueeze(0)
    return frames

def _uniform_frame_indices(total_frames: int, num_samples: int):
    if num_samples <= 0:
        return []
    if total_frames <= 0:
        return [0 for _ in range(num_samples)]
    if total_frames == 1:
        return [0 for _ in range(num_samples)]
    if num_samples == 1:
        return [int(total_frames // 2)]
    idx = np.linspace(0, total_frames - 1, num_samples)
    return np.round(idx).astype(int).tolist()

def _uniform_frame_positions(total_frames: int, num_samples: int):
    if num_samples <= 0:
        return []
    if total_frames <= 0:
        return [0.0 for _ in range(num_samples)]
    if total_frames == 1:
        return [0.0 for _ in range(num_samples)]
    if num_samples == 1:
        return [0.5 * float(total_frames - 1)]
    return np.linspace(0, total_frames - 1, num_samples).tolist()

def _context_frame_triplets(total_frames: int, num_samples: int, context_ratio: float):
    positions = _uniform_frame_positions(total_frames, num_samples)
    if not positions:
        return []
    if total_frames <= 0:
        return [[0, 0, 0] for _ in range(num_samples)]
    if len(positions) >= 2:
        interval = float(positions[1] - positions[0])
    elif total_frames > 1:
        interval = float(total_frames - 1)
    else:
        interval = 0.0
    offset = max(0.0, float(context_ratio)) * interval
    max_index = total_frames - 1
    triplets = []
    for pos in positions:
        center = int(round(pos))
        center = max(0, min(max_index, center))
        if offset <= 0.0:
            left = center
            right = center
        else:
            left = int(round(max(0.0, pos - offset)))
            right = int(round(min(float(max_index), pos + offset)))
        triplets.append([left, center, right])
    return triplets


_TEMPORAL_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _coerce_temporal_position(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        match = _TEMPORAL_NUMBER_RE.search(str(value))
        if match is None:
            return default
        try:
            return float(match.group(0))
        except ValueError:
            return default


def parse_temporal_sample_id(sample_id):
    sample_text = str(sample_id)
    if "_" in sample_text:
        base_id, clip_idx = sample_text.rsplit("_", 1)
    else:
        base_id, clip_idx = sample_text, ""
    clip_pos = _coerce_temporal_position(clip_idx, default=0.0)
    return base_id, clip_idx, float(clip_pos)


def build_temporal_metadata(sample_id, label=None, group_id=0):
    base_id, clip_idx, clip_pos = parse_temporal_sample_id(sample_id)
    label = label or {}
    temporal_pos = clip_pos
    if isinstance(label, dict):
        for key in ("start", "start_time", "start_sec", "begin", "begin_time"):
            if key in label:
                temporal_pos = _coerce_temporal_position(label.get(key), default=clip_pos)
                break
    return {
        "base_id": base_id,
        "clip_idx": clip_idx,
        "temporal_pos": float(temporal_pos),
        "temporal_group_id": int(group_id),
    }


def build_temporal_context_index(
    ids,
    temporal_metadata_by_id,
    radius=0,
    mode="neighbors",
    seed=0,
):
    """Build fixed-width masked-center temporal windows.

    Invalid boundary slots point back to the center sample but carry
    ``valid=False``.  This keeps default collation rectangular without letting
    the center leak into the context prior.  ``shuffled`` preserves the valid
    slot and relative-position pattern while replacing its content with a clip
    from another temporal group.
    """
    radius = int(radius)
    mode = str(mode).lower()
    if radius < 0:
        raise ValueError(f"temporal context radius must be non-negative, got {radius}")
    if mode not in {"neighbors", "shuffled"}:
        raise ValueError(
            "temporal context mode must be 'neighbors' or 'shuffled', "
            f"got {mode!r}"
        )
    if radius == 0:
        return {idx: [] for idx in range(len(ids))}

    groups = {}
    for idx, sample_id in enumerate(ids):
        metadata = temporal_metadata_by_id[sample_id]
        group_id = int(metadata["temporal_group_id"])
        groups.setdefault(group_id, []).append(idx)
    for group_indices in groups.values():
        group_indices.sort(
            key=lambda idx: (
                float(temporal_metadata_by_id[ids[idx]]["temporal_pos"]),
                str(ids[idx]),
            )
        )

    group_by_index = {}
    position_in_group = {}
    for group_id, group_indices in groups.items():
        for group_pos, idx in enumerate(group_indices):
            group_by_index[idx] = group_id
            position_in_group[idx] = group_pos

    rng = random.Random(int(seed))
    all_indices = list(range(len(ids)))
    shuffled_candidates = {
        group_id: [idx for idx in all_indices if group_by_index[idx] != group_id]
        for group_id in groups
    }
    offsets = list(range(-radius, 0)) + list(range(1, radius + 1))
    context_index = {}
    for center_idx, center_id in enumerate(ids):
        group_id = group_by_index[center_idx]
        group_indices = groups[group_id]
        center_group_pos = position_in_group[center_idx]
        center_temporal_pos = float(
            temporal_metadata_by_id[center_id]["temporal_pos"]
        )
        rows = []
        for offset in offsets:
            neighbor_group_pos = center_group_pos + offset
            valid = 0 <= neighbor_group_pos < len(group_indices)
            neighbor_idx = (
                group_indices[neighbor_group_pos] if valid else center_idx
            )
            if valid:
                neighbor_id = ids[neighbor_idx]
                relative_pos = float(
                    temporal_metadata_by_id[neighbor_id]["temporal_pos"]
                ) - center_temporal_pos
                if mode == "shuffled":
                    candidates = shuffled_candidates[group_id]
                    if not candidates:
                        valid = False
                        neighbor_idx = center_idx
                    else:
                        neighbor_idx = candidates[rng.randrange(len(candidates))]
            else:
                relative_pos = 0.0
            rows.append(
                {
                    "index": int(neighbor_idx),
                    "valid": bool(valid),
                    "relative_pos": float(relative_pos),
                }
            )
        context_index[center_idx] = rows
    return context_index


def _sample_id_from_row(row, fallback):
    for key in ("id", "sample_id", "name", "video_id", "utterance_id"):
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return str(fallback)

def _stack_frames(frames, transform, num_frames, fallback_size=112):
    if not frames:
        frames = [Image.new("RGB", (fallback_size, fallback_size), (0, 0, 0)) for _ in range(num_frames)]
    tensors = []
    for fr in frames:
        if isinstance(fr, np.ndarray):
            fr = Image.fromarray(fr.astype(np.uint8))
        if transform is not None:
            fr_t = transform(fr)
        else:
            fr_t = torch.from_numpy(np.array(fr)).permute(2, 0, 1).float() / 255.0
        tensors.append(fr_t)
    return torch.stack(tensors, dim=0)

def _stack_frame_triplets(frame_triplets, transform, num_frames, fallback_size=112):
    if not frame_triplets:
        blank = Image.new("RGB", (fallback_size, fallback_size), (0, 0, 0))
        frame_triplets = [[blank, blank, blank] for _ in range(num_frames)]
    if len(frame_triplets) < num_frames and frame_triplets:
        frame_triplets = list(frame_triplets) + [frame_triplets[-1]] * (num_frames - len(frame_triplets))
    stacked = []
    for triplet in frame_triplets[:num_frames]:
        if not triplet:
            blank = Image.new("RGB", (fallback_size, fallback_size), (0, 0, 0))
            triplet = [blank, blank, blank]
        triplet = list(triplet[:3])
        while len(triplet) < 3:
            triplet.append(triplet[-1])
        stacked.append(_stack_frames(triplet, transform, num_frames=3, fallback_size=fallback_size))
    return torch.stack(stacked, dim=0)

def _pretokenize_texts(tokenizer, texts, max_length: int, batch_size: int = 256):
    texts = [str(text or "") for text in texts]
    if not texts:
        return []
    if tokenizer is None:
        zeros_ids = torch.zeros(max_length, dtype=torch.long)
        zeros_mask = torch.zeros(max_length, dtype=torch.long)
        return [(zeros_ids.clone(), zeros_mask.clone()) for _ in texts]

    tokenized = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_input_ids = encoded["input_ids"]
        batch_attention_mask = encoded["attention_mask"]
        for row_idx in range(batch_input_ids.shape[0]):
            tokenized.append((
                batch_input_ids[row_idx].clone(),
                batch_attention_mask[row_idx].clone(),
            ))
    return tokenized

def _load_openface_array(npy_path: str, sample_id: str):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"OpenFace face track not found for sample '{sample_id}': {npy_path}"
        )
    try:
        return np.load(npy_path, mmap_mode="r")
    except TypeError:
        return np.load(npy_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load OpenFace face track for sample '{sample_id}': {npy_path}"
        ) from exc


def preprocess_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
    max_seconds: float,
    frame_ms: float,
    hop_ms: float,
    denoise: bool = True,
):
    waveform = _to_mono(waveform).float()
    if waveform.numel() == 0:
        raise ValueError("Audio waveform is empty")
    orig_sr = int(orig_sr) if int(orig_sr) > 0 else int(target_sr)
    waveform = _resample_waveform(waveform, orig_sr, target_sr)
    if denoise:
        n_fft = int(round(target_sr * 0.025))
        hop_length = int(round(target_sr * 0.01))
        waveform = _spectral_denoise(waveform, n_fft=n_fft, hop_length=hop_length, noise_frames=6)
    frame_len = int(round(target_sr * frame_ms / 1000.0))
    hop_len = int(round(target_sr * hop_ms / 1000.0))
    frames = _frame_waveform(waveform, frame_len, hop_len)
    max_len = int(round(target_sr * max_seconds))
    if max_len <= 0:
        max_len = waveform.numel()
    raw_len = waveform.numel()
    if raw_len < max_len:
        waveform = F.pad(waveform, (0, max_len - raw_len))
    else:
        waveform = waveform[:max_len]
        raw_len = max_len
    attention_mask = torch.zeros(max_len, dtype=torch.long)
    if raw_len > 0:
        attention_mask[:raw_len] = 1
    return waveform, attention_mask, frames

def _load_audio_from_video(video_path: str):
    if not video_path or not os.path.exists(str(video_path)):
        raise FileNotFoundError(f"Video file not found for audio extraction: {video_path}")
    try:
        _, audio, info = read_video(video_path, pts_unit="sec")
        if audio is None or audio.numel() == 0:
            raise ValueError(f"No audio stream found in video: {video_path}")
        sr = int(info.get("audio_fps", 0) or info.get("audio_sample_rate", 0) or 0)
        if sr <= 0:
            raise ValueError(f"Invalid audio sample rate from video: {video_path}")
        return audio, sr
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio from video: {video_path}") from exc

def _load_audio_from_path(audio_path: str):
    if not audio_path or not os.path.exists(str(audio_path)):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = os.path.splitext(str(audio_path))[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return _load_audio_from_video(audio_path)
    if _HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform, int(sr)
        except Exception:
            pass
    if ext in {".wav", ".flac"}:
        try:
            sr, waveform = wavfile.read(audio_path)
            if waveform is None:
                return torch.zeros(0), 0
            waveform = np.asarray(waveform)
            if np.issubdtype(waveform.dtype, np.integer):
                max_abs = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
                waveform = waveform.astype(np.float32) / float(max_abs)
            else:
                waveform = waveform.astype(np.float32, copy=False)
            if waveform.ndim == 1:
                waveform = torch.from_numpy(waveform)
            else:
                waveform = torch.from_numpy(waveform)
            sr = int(sr)
            if sr <= 0:
                raise ValueError(f"Invalid audio sample rate in file: {audio_path}")
            if waveform.numel() == 0:
                raise ValueError(f"Audio file produced empty waveform: {audio_path}")
            return waveform, sr
        except Exception:
            pass
    try:
        _, audio, info = read_video(audio_path, pts_unit="sec")
        if audio is None or audio.numel() == 0:
            raise ValueError(f"Audio file produced empty waveform: {audio_path}")
        sr = int(info.get("audio_fps", 0) or 0)
        if sr <= 0:
            raise ValueError(f"Invalid audio sample rate in file: {audio_path}")
        return audio, sr
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file: {audio_path}") from exc

class MultimodalEmotionDataset(Dataset):
    def __init__(
        self,
        csv_file,
        split='train',
        transform=None,
        tokenizer_path="/path/to/models/google-bert_bert-base-uncased",
        max_length=128,
        local_files_only=True,
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=1,
    ):
        self.csv_file = csv_file
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.data = []
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)

        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'split' in row and row['split'] != split:
                        continue
                    self.data.append(row)
        else:
            print(f"Warning: CSV file {csv_file} not found. Initializing empty dataset.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [row.get("text", "") for row in self.data],
            self.max_length,
        )
        self.sample_ids = [_sample_id_from_row(row, idx) for idx, row in enumerate(self.data)]
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted({parse_temporal_sample_id(sample_id)[0] for sample_id in self.sample_ids})
            )
        }
        self.temporal_metadata = [
            build_temporal_metadata(sample_id, row, group_id=group_ids[parse_temporal_sample_id(sample_id)[0]])
            for sample_id, row in zip(self.sample_ids, self.data)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sample_id = self.sample_ids[idx]
        temporal_metadata = self.temporal_metadata[idx]

        image_path = row.get('image_path', '')
        if not image_path or not os.path.exists(str(image_path)):
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            try:
                with Image.open(image_path) as pil_image:
                    image = pil_image.convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if self.num_frames > 1:
            image = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        input_ids, attention_mask = self.encoded_text[idx]

        raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
        if raw_valence is None or raw_valence == "":
            raise ValueError("raw_valence is required for MultimodalEmotionDataset")
        raw_valence = float(raw_valence)
        intensity = _valence_to_intensity(raw_valence)
        polarity = _valence_to_polarity(raw_valence)

        audio_path = row.get("audio_path", "")
        waveform, sr = _load_audio_from_path(str(audio_path))
        waveform, audio_mask, _ = preprocess_audio(
            waveform,
            orig_sr=sr,
            target_sr=self.audio_sample_rate,
            max_seconds=self.audio_max_seconds,
            frame_ms=self.audio_frame_ms,
            hop_ms=self.audio_hop_ms,
            denoise=self.audio_denoise,
        )

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'raw_valence': torch.tensor(raw_valence, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32),
            'polarity': torch.tensor(polarity, dtype=torch.long),
            'id': sample_id,
            'base_id': temporal_metadata["base_id"],
            'clip_idx': temporal_metadata["clip_idx"],
            'temporal_pos': torch.tensor(temporal_metadata["temporal_pos"], dtype=torch.float32),
            'temporal_group_id': torch.tensor(temporal_metadata["temporal_group_id"], dtype=torch.long),
            'audio_values': waveform,
            'audio_attention_mask': audio_mask,
        }

def _valence_to_7class(val):
    v = float(val)
    if v >= 0:
        rounded = math.floor(v + 0.5)
    else:
        rounded = math.ceil(v - 0.5)
    rounded = max(-3, min(3, int(rounded)))
    return rounded + 3

def _valence_to_binary(val):
    if val >= 0:
        return 1
    return 0

def _valence_to_intensity(val):
    intensity = abs(float(val))
    if intensity < 0:
        intensity = 0.0
    if intensity > 3.0:
        intensity = 3.0
    return intensity

def _valence_to_polarity(val):
    v = float(val)
    if v < 0:
        return 0
    return 1

class CMUMOSEIProcessDataset(Dataset):
    def __init__(
        self,
        root="/path/to/datasets/cmumosei-process",
        split="train",
        transform=None,
        tokenizer_path="/path/to/models/google-bert_bert-base-uncased",
        max_length=128,
        local_files_only=True,
        frame_policy="middle",
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=4,
        openface_face_dir=None,
        vit_context_ratio=0.0,
        embedding_cache_root=None,
        temporal_context_radius=0,
        temporal_context_mode="neighbors",
        temporal_context_seed=0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.frame_policy = frame_policy
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)
        self.openface_face_dir = openface_face_dir or os.path.join(root, "openface_face")
        self.vit_context_ratio = float(vit_context_ratio)
        self.temporal_context_radius = int(temporal_context_radius)
        self.temporal_context_mode = str(temporal_context_mode).lower()
        self.temporal_context_seed = int(temporal_context_seed)
        self.embedding_cache = None

        label_npz = np.load(os.path.join(root, "label.npz"), allow_pickle=True)
        split_key = {"train": "train_corpus", "val": "val_corpus", "test": "test_corpus"}[split]
        self.labels = label_npz[split_key].item()

        trans_path = os.path.join(root, "transcription-engchi-polish.csv")
        self.text_by_id = {}
        with open(trans_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0] == "id":
                    continue
                if len(row) >= 3:
                    self.text_by_id[row[0]] = row[2]
                elif len(row) == 2:
                    self.text_by_id[row[0]] = row[1]
                else:
                    self.text_by_id[row[0]] = ""

        self.ids = sorted(self.labels.keys())
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted({parse_temporal_sample_id(sample_id)[0] for sample_id in self.ids})
            )
        }
        self.temporal_metadata_by_id = {
            sample_id: build_temporal_metadata(
                sample_id,
                self.labels.get(sample_id, {}),
                group_id=group_ids[parse_temporal_sample_id(sample_id)[0]],
            )
            for sample_id in self.ids
        }
        self.temporal_context_index = build_temporal_context_index(
            self.ids,
            self.temporal_metadata_by_id,
            radius=self.temporal_context_radius,
            mode=self.temporal_context_mode,
            seed=self.temporal_context_seed,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [self.text_by_id.get(sample_id, "") for sample_id in self.ids],
            self.max_length,
        )
        if embedding_cache_root:
            self.embedding_cache = MultimodalEmbeddingCache(embedding_cache_root, split)
            _validate_embedding_cache(self.embedding_cache, self.ids, split)

    def __len__(self):
        return len(self.ids)

    def _load_video_frame(self, video_path):
        try:
            v, _, _ = read_video(video_path, pts_unit="sec")
            if v.numel() == 0:
                return Image.new("RGB", (224, 224), (0, 0, 0))
            if self.frame_policy == "first":
                frame = v[0]
            elif self.frame_policy == "last":
                frame = v[-1]
            else:
                frame = v[v.shape[0] // 2]
            frame = frame.numpy()
            return Image.fromarray(frame).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (0, 0, 0))

    def _load_openface_frames(self, sample_id: str):
        npy_path = os.path.join(self.openface_face_dir, f"{sample_id}.npy")
        arr = _load_openface_array(npy_path, sample_id)
        if arr.ndim != 4:
            raise ValueError(
                f"Invalid OpenFace face track shape for sample '{sample_id}': "
                f"expected 4D array, got {tuple(arr.shape)} from {npy_path}"
            )
        total = int(arr.shape[0])
        if total <= 0:
            raise ValueError(
                f"Empty OpenFace face track for sample '{sample_id}': {npy_path}"
            )
        indices = _uniform_frame_indices(total, self.num_frames)
        frames = [arr[i] for i in indices if 0 <= i < total]
        if not frames:
            raise ValueError(
                f"Could not sample any OpenFace frames for sample '{sample_id}' from {npy_path}"
            )
        if len(frames) < self.num_frames:
            pad = self.num_frames - len(frames)
            if frames:
                frames.extend([frames[-1]] * pad)
        return frames

    def _load_openface_frame_triplets(self, sample_id: str):
        npy_path = os.path.join(self.openface_face_dir, f"{sample_id}.npy")
        arr = _load_openface_array(npy_path, sample_id)
        if arr.ndim != 4:
            raise ValueError(
                f"Invalid OpenFace face track shape for sample '{sample_id}': "
                f"expected 4D array, got {tuple(arr.shape)} from {npy_path}"
            )
        total = int(arr.shape[0])
        if total <= 0:
            raise ValueError(
                f"Empty OpenFace face track for sample '{sample_id}': {npy_path}"
            )
        triplet_indices = _context_frame_triplets(total, self.num_frames, self.vit_context_ratio)
        triplets = []
        for left_idx, center_idx, right_idx in triplet_indices:
            triplets.append([arr[left_idx], arr[center_idx], arr[right_idx]])
        if not triplets:
            raise ValueError(
                f"Could not sample any OpenFace frame triplets for sample '{sample_id}' from {npy_path}"
            )
        if len(triplets) < self.num_frames:
            pad = self.num_frames - len(triplets)
            triplets.extend([triplets[-1]] * pad)
        return triplets

    def _get_single_item(self, idx):
        sample_id = self.ids[idx]
        temporal_metadata = self.temporal_metadata_by_id[sample_id]
        video_path = os.path.join(self.root, "subvideo_new", f"{sample_id}.mp4")
        audio_path = os.path.join(self.root, "subaudio", f"{sample_id}.wav")
        input_ids, attention_mask = self.encoded_text[idx]
        val = float(self.labels[sample_id]["val"])
        intensity = _valence_to_intensity(val)
        polarity = _valence_to_polarity(val)

        cached = self.embedding_cache.get(sample_id) if self.embedding_cache is not None else None
        if cached is not None:
            item = {
                "image": torch.empty(0, dtype=torch.float32),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "raw_valence": torch.tensor(val, dtype=torch.float32),
                "intensity": torch.tensor(intensity, dtype=torch.float32),
                "polarity": torch.tensor(polarity, dtype=torch.long),
                "id": sample_id,
                "base_id": temporal_metadata["base_id"],
                "clip_idx": temporal_metadata["clip_idx"],
                "temporal_pos": torch.tensor(temporal_metadata["temporal_pos"], dtype=torch.float32),
                "temporal_group_id": torch.tensor(temporal_metadata["temporal_group_id"], dtype=torch.long),
                "audio_values": torch.empty(0, dtype=torch.float32),
                "audio_attention_mask": torch.empty(0, dtype=torch.long),
            }
            item.update(cached)
            return item

        if self.vit_context_ratio > 0.0:
            frame_triplets = self._load_openface_frame_triplets(sample_id)
            image = _stack_frame_triplets(frame_triplets, self.transform, self.num_frames, fallback_size=112)
        else:
            frames = self._load_openface_frames(sample_id)
            image = _stack_frames(frames, self.transform, self.num_frames, fallback_size=112)

        waveform, sr = _load_audio_from_path(audio_path)

        waveform, audio_mask, _ = preprocess_audio(
            waveform,
            orig_sr=sr,
            target_sr=self.audio_sample_rate,
            max_seconds=self.audio_max_seconds,
            frame_ms=self.audio_frame_ms,
            hop_ms=self.audio_hop_ms,
            denoise=self.audio_denoise,
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_valence": torch.tensor(val, dtype=torch.float32),
            "intensity": torch.tensor(intensity, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.long),
            "id": sample_id,
            "base_id": temporal_metadata["base_id"],
            "clip_idx": temporal_metadata["clip_idx"],
            "temporal_pos": torch.tensor(temporal_metadata["temporal_pos"], dtype=torch.float32),
            "temporal_group_id": torch.tensor(temporal_metadata["temporal_group_id"], dtype=torch.long),
            "audio_values": waveform,
            "audio_attention_mask": audio_mask,
        }

    def __getitem__(self, idx):
        item = self._get_single_item(idx)
        if self.temporal_context_radius <= 0:
            return item

        context_rows = self.temporal_context_index[idx]
        if not context_rows:
            raise RuntimeError(
                "temporal_context_radius is positive but no context slots were built"
            )
        context_items = [
            self._get_single_item(row["index"])
            for row in context_rows
        ]
        stack_keys = (
            "image",
            "input_ids",
            "attention_mask",
            "audio_values",
            "audio_attention_mask",
            "raw_valence",
            "cached_vision",
            "cached_text",
            "cached_audio",
            "cached_text_mask",
            "cached_audio_mask",
        )
        for key in stack_keys:
            if all(key in context_item for context_item in context_items):
                item[f"tcif_context_{key}"] = torch.stack(
                    [context_item[key] for context_item in context_items],
                    dim=0,
                )
        item["tcif_context_valid_mask"] = torch.tensor(
            [row["valid"] for row in context_rows],
            dtype=torch.bool,
        )
        item["tcif_context_relative_pos"] = torch.tensor(
            [row["relative_pos"] for row in context_rows],
            dtype=torch.float32,
        )
        return item

class CMUMOSIProcessDataset(Dataset):
    def __init__(
        self,
        root="/path/to/datasets/cmumosi-process",
        split="train",
        transform=None,
        tokenizer_path="/path/to/models/google-bert_bert-base-uncased",
        max_length=128,
        local_files_only=True,
        frame_policy="middle",
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=4,
        openface_face_dir=None,
        vit_context_ratio=0.0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.frame_policy = frame_policy
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)
        self.openface_face_dir = openface_face_dir or os.path.join(root, "openface_face")
        self.vit_context_ratio = float(vit_context_ratio)

        label_npz = np.load(os.path.join(root, "label.npz"), allow_pickle=True)
        split_key = {"train": "train_corpus", "val": "val_corpus", "test": "test_corpus"}[split]
        self.labels = label_npz[split_key].item()

        trans_path = os.path.join(root, "transcription-engchi-polish.csv")
        self.text_by_id = {}
        with open(trans_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0] in {"id", "name"}:
                    continue
                if len(row) >= 3:
                    self.text_by_id[row[0]] = row[2]
                elif len(row) == 2:
                    self.text_by_id[row[0]] = row[1]
                else:
                    self.text_by_id[row[0]] = ""

        self.ids = sorted(self.labels.keys())
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted({parse_temporal_sample_id(sample_id)[0] for sample_id in self.ids})
            )
        }
        self.temporal_metadata_by_id = {
            sample_id: build_temporal_metadata(
                sample_id,
                self.labels.get(sample_id, {}),
                group_id=group_ids[parse_temporal_sample_id(sample_id)[0]],
            )
            for sample_id in self.ids
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [self.text_by_id.get(sample_id, "") for sample_id in self.ids],
            self.max_length,
        )

    def __len__(self):
        return len(self.ids)

    def _load_openface_frames(self, sample_id: str):
        npy_path = os.path.join(self.openface_face_dir, f"{sample_id}.npy")
        arr = _load_openface_array(npy_path, sample_id)
        if arr.ndim != 4:
            raise ValueError(
                f"Invalid OpenFace face track shape for sample '{sample_id}': "
                f"expected 4D array, got {tuple(arr.shape)} from {npy_path}"
            )
        total = int(arr.shape[0])
        if total <= 0:
            raise ValueError(
                f"Empty OpenFace face track for sample '{sample_id}': {npy_path}"
            )
        indices = _uniform_frame_indices(total, self.num_frames)
        frames = [arr[i] for i in indices if 0 <= i < total]
        if not frames:
            raise ValueError(
                f"Could not sample any OpenFace frames for sample '{sample_id}' from {npy_path}"
            )
        if len(frames) < self.num_frames:
            pad = self.num_frames - len(frames)
            if frames:
                frames.extend([frames[-1]] * pad)
        return frames

    def _load_openface_frame_triplets(self, sample_id: str):
        npy_path = os.path.join(self.openface_face_dir, f"{sample_id}.npy")
        arr = _load_openface_array(npy_path, sample_id)
        if arr.ndim != 4:
            raise ValueError(
                f"Invalid OpenFace face track shape for sample '{sample_id}': "
                f"expected 4D array, got {tuple(arr.shape)} from {npy_path}"
            )
        total = int(arr.shape[0])
        if total <= 0:
            raise ValueError(
                f"Empty OpenFace face track for sample '{sample_id}': {npy_path}"
            )
        triplet_indices = _context_frame_triplets(total, self.num_frames, self.vit_context_ratio)
        triplets = []
        for left_idx, center_idx, right_idx in triplet_indices:
            triplets.append([arr[left_idx], arr[center_idx], arr[right_idx]])
        if not triplets:
            raise ValueError(
                f"Could not sample any OpenFace frame triplets for sample '{sample_id}' from {npy_path}"
            )
        if len(triplets) < self.num_frames:
            pad = self.num_frames - len(triplets)
            triplets.extend([triplets[-1]] * pad)
        return triplets

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        temporal_metadata = self.temporal_metadata_by_id[sample_id]
        video_path = os.path.join(self.root, "subvideo", f"{sample_id}.mp4")
        audio_path = os.path.join(self.root, "subaudio", f"{sample_id}.wav")

        if self.vit_context_ratio > 0.0:
            frame_triplets = self._load_openface_frame_triplets(sample_id)
            image = _stack_frame_triplets(frame_triplets, self.transform, self.num_frames, fallback_size=112)
        else:
            frames = self._load_openface_frames(sample_id)
            image = _stack_frames(frames, self.transform, self.num_frames, fallback_size=112)

        input_ids, attention_mask = self.encoded_text[idx]

        val = float(self.labels[sample_id]["val"])
        intensity = _valence_to_intensity(val)
        polarity = _valence_to_polarity(val)

        waveform, sr = _load_audio_from_path(audio_path)
        waveform, audio_mask, _ = preprocess_audio(
            waveform,
            orig_sr=sr,
            target_sr=self.audio_sample_rate,
            max_seconds=self.audio_max_seconds,
            frame_ms=self.audio_frame_ms,
            hop_ms=self.audio_hop_ms,
            denoise=self.audio_denoise,
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_valence": torch.tensor(val, dtype=torch.float32),
            "intensity": torch.tensor(intensity, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.long),
            "id": sample_id,
            "base_id": temporal_metadata["base_id"],
            "clip_idx": temporal_metadata["clip_idx"],
            "temporal_pos": torch.tensor(temporal_metadata["temporal_pos"], dtype=torch.float32),
            "temporal_group_id": torch.tensor(temporal_metadata["temporal_group_id"], dtype=torch.long),
            "audio_values": waveform,
            "audio_attention_mask": audio_mask,
        }
