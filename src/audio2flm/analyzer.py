from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter

from .flm import NoteEvent


@dataclass
class AnalysisConfig:
    bpm: float = 120.0
    fmin: str = "C2"
    fmax: str = "C7"
    frame_length: int = 2048
    hop_length: int = 256
    min_note_ms: float = 45.0
    max_gap_ms: float = 35.0
    pitch_change_semitones: float = 0.60
    stable_cents: float = 38.0
    slide_min_semitones: float = 0.80
    slide_max_ms: float = 280.0
    onset_split_strength: float = 0.55


@dataclass
class Segment:
    start: int
    end: int
    midi: float


def _load(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    if len(y) == 0:
        raise ValueError("Áudio vazio.")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), int(sr)


def _track_pitch(y, sr, cfg: AnalysisConfig):
    fmin = librosa.note_to_hz(cfg.fmin)
    fmax = librosa.note_to_hz(cfg.fmax)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
            fill_na=np.nan,
        )
    except Exception:
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )
        voiced_flag = np.isfinite(f0)

    rms = librosa.feature.rms(
        y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length
    )[0]
    n = min(len(f0), len(rms))
    f0 = f0[:n]
    rms = rms[:n]
    voiced_flag = np.asarray(voiced_flag[:n], dtype=bool)
    midi = librosa.hz_to_midi(f0)

    valid = np.isfinite(midi) & voiced_flag
    if valid.any():
        filled = midi.copy()
        idx = np.arange(n)
        filled[~valid] = np.interp(idx[~valid], idx[valid], filled[valid])
        filled = median_filter(filled, size=5, mode="nearest")
        midi[valid] = filled[valid]

    nonzero = rms[rms > 1e-7]
    floor = np.percentile(nonzero, 15) if len(nonzero) else 0.0
    silence = max(0.003, float(floor) * 0.65)
    valid &= rms > silence
    times = librosa.frames_to_time(
        np.arange(n), sr=sr, hop_length=cfg.hop_length
    )
    return times, midi, rms, valid


def _bridge_short_gaps(valid, max_gap_frames):
    bridged = valid.copy()
    i = 0
    while i < len(bridged):
        if bridged[i]:
            i += 1
            continue
        j = i
        while j < len(bridged) and not bridged[j]:
            j += 1
        if i > 0 and j < len(bridged) and (j - i) <= max_gap_frames:
            bridged[i:j] = True
        i = j
    return bridged


def _segments(midi, rms, valid, sr, cfg):
    hop_ms = cfg.hop_length * 1000.0 / sr
    valid = _bridge_short_gaps(
        valid, max(1, round(cfg.max_gap_ms / hop_ms))
    )
    min_frames = max(2, round(cfg.min_note_ms / hop_ms))
    raw = []
    i = 0

    while i < len(valid):
        if not valid[i] or not np.isfinite(midi[i]):
            i += 1
            continue
        start = i
        values = [float(midi[i])]
        j = i + 1
        while j < len(valid) and valid[j] and np.isfinite(midi[j]):
            median = float(np.median(values[-10:]))
            if abs(float(midi[j]) - median) >= cfg.pitch_change_semitones:
                look = midi[j : min(len(midi), j + 3)]
                look = look[np.isfinite(look)]
                if (
                    len(look) >= 2
                    and abs(float(np.median(look)) - median)
                    >= cfg.pitch_change_semitones
                ):
                    break
            values.append(float(midi[j]))
            j += 1

        if j - start >= min_frames:
            region = midi[start:j]
            region = region[np.isfinite(region)]
            raw.append(Segment(start, j, float(np.median(region))))
        i = max(j, i + 1)

    merged = []
    for segment in raw:
        if merged:
            previous = merged[-1]
            gap = segment.start - previous.end
            if round(segment.midi) == round(previous.midi) and gap <= 2:
                previous.end = segment.end
                region = midi[previous.start : previous.end]
                region = region[np.isfinite(region)]
                if len(region):
                    previous.midi = float(np.median(region))
                continue
        merged.append(segment)
    return merged, valid


def _fine_pitch(midi_value: float, key: int) -> int:
    cents = (midi_value - key) * 100.0
    return int(round(120 + np.clip(cents, -120, 120)))


def analyze_audio(path: str, cfg: AnalysisConfig):
    y, sr = _load(path)
    times, midi, rms, valid = _track_pitch(y, sr, cfg)
    segments, bridged = _segments(midi, rms, valid, sr, cfg)
    if not segments:
        raise ValueError("Nenhuma nota vocal confiável foi detectada.")

    sec_to_beat = cfg.bpm / 60.0
    events = []
    debug = []

    for index, segment in enumerate(segments):
        start_t = float(times[segment.start])
        end_index = min(len(times) - 1, max(segment.start, segment.end - 1))
        end_t = float(times[end_index] + cfg.hop_length / sr)
        key = int(np.clip(round(segment.midi), 0, 127))
        local_rms = rms[segment.start : segment.end]
        velocity = int(
            np.clip(
                round(
                    55
                    + 73
                    * (
                        float(np.median(local_rms))
                        / max(1e-6, float(np.percentile(rms, 95)))
                    )
                ),
                1,
                128,
            )
        )
        carrier = NoteEvent(
            start_t * sec_to_beat,
            max(1e-4, (end_t - start_t) * sec_to_beat),
            key,
            velocity,
            64,
            _fine_pitch(segment.midi, key),
            False,
        )
        events.append(carrier)
        debug.append(
            {
                "type": "note",
                "start_s": start_t,
                "end_s": end_t,
                "midi": segment.midi,
                "key": key,
            }
        )

        if index + 1 >= len(segments):
            continue
        next_segment = segments[index + 1]
        if abs(next_segment.midi - segment.midi) < cfg.slide_min_semitones:
            continue

        a = max(segment.start, segment.end - 2)
        b = min(len(bridged), next_segment.start + 3)
        continuous = b > a and float(np.mean(bridged[a:b])) >= 0.75
        if not continuous:
            continue

        boundary = max(segment.start, min(segment.end - 1, next_segment.start))
        pre = (
            float(np.median(rms[max(segment.start, boundary - 3) : boundary + 1]))
            if boundary >= segment.start
            else 0.0
        )
        valley = (
            float(np.min(rms[max(0, boundary - 1) : min(len(rms), boundary + 2)]))
            if len(rms)
            else 0.0
        )
        strong_onset_split = pre > 1e-6 and valley / pre < cfg.onset_split_strength
        if strong_onset_split:
            continue

        transition_start = max(segment.start, segment.end - 2)
        transition_end = min(next_segment.end, next_segment.start + 5)
        transition_ms = (
            (transition_end - transition_start)
            * cfg.hop_length
            * 1000.0
            / sr
        )
        if transition_ms > cfg.slide_max_ms:
            continue

        slide_start = float(times[transition_start])
        slide_end = float(
            times[min(len(times) - 1, max(transition_start, transition_end - 1))]
            + cfg.hop_length / sr
        )
        target_key = int(np.clip(round(next_segment.midi), 0, 127))
        events.append(
            NoteEvent(
                slide_start * sec_to_beat,
                max(1e-4, (slide_end - slide_start) * sec_to_beat),
                target_key,
                velocity,
                64,
                _fine_pitch(next_segment.midi, target_key),
                True,
            )
        )
        carrier.duration_beat = max(
            carrier.duration_beat, (slide_end - start_t) * sec_to_beat
        )
        debug.append(
            {
                "type": "slide",
                "start_s": slide_start,
                "end_s": slide_end,
                "from": segment.midi,
                "to": next_segment.midi,
                "key": target_key,
            }
        )

    return events, {
        "sample_rate": sr,
        "duration_s": len(y) / sr,
        "frames": len(times),
        "segments": len(segments),
        "events": debug,
    }
