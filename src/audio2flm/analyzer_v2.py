from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter

from .flm import NoteEvent


@dataclass
class AnalysisConfigV2:
    bpm: float = 120.0
    fmin: str = "C2"
    fmax: str = "C7"
    analysis_sr: int = 22050
    frame_length: int = 1024
    hop_length: int = 128
    pitch_offset: float = 0.0
    grid_beat: float = 1.0 / 16.0
    min_note_ms: float = 30.0
    max_gap_ms: float = 28.0
    stable_semitones: float = 0.42
    slide_step_semitones: float = 0.55
    slide_min_delta: float = 0.70
    slide_max_gap_ms: float = 90.0
    velocity: int = 102


def _snap(value: float, grid: float) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def _load(path: str, sr: int):
    y, actual_sr = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("Áudio vazio.")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), int(actual_sr)


def _pitch_track(y: np.ndarray, sr: int, cfg: AnalysisConfigV2):
    fmin = librosa.note_to_hz(cfg.fmin)
    fmax = librosa.note_to_hz(cfg.fmax)

    # YIN is substantially faster than pYIN for long FNF vocal stems and,
    # after median smoothing + energy gating, proved more stable on the first
    # real calibration pair.
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    midi = librosa.hz_to_midi(f0) + cfg.pitch_offset
    rms = librosa.feature.rms(
        y=y,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )[0]
    n = min(len(midi), len(rms))
    midi = midi[:n]
    rms = rms[:n]

    finite = np.isfinite(midi)
    if finite.any():
        filled = midi.copy()
        idx = np.arange(n)
        filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
        filled = median_filter(filled, size=5, mode="nearest")
        midi[finite] = filled[finite]

    nz = rms[rms > 1e-7]
    if len(nz):
        # A percentile gate is more robust on compressed MP3 vocal exports than
        # the v0.1 fixed floor.
        gate = max(0.004, float(np.percentile(nz, 22)) * 1.12)
    else:
        gate = 0.004
    voiced = finite & (rms >= gate)

    return midi, rms, voiced


def _bridge(voiced: np.ndarray, max_frames: int) -> np.ndarray:
    out = voiced.copy()
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and (j - i) <= max_frames:
            out[i:j] = True
        i = j
    return out


def _region_bounds(voiced: np.ndarray):
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        while j < len(voiced) and voiced[j]:
            j += 1
        yield i, j
        i = j


def _compress_curve(values: np.ndarray, start: int, end: int, threshold: float):
    """Return pitch anchors while preserving real bends instead of stair-stepping.

    Anchors are emitted only when the smoothed curve moves enough from the last
    anchor.  The writer then represents later anchors as FLM slide notes.
    """
    anchors: list[tuple[int, float]] = []
    finite = values[start:end]
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return anchors

    first_pitch = float(np.median(values[start : min(end, start + 4)]))
    anchors.append((start, first_pitch))
    anchor_pitch = first_pitch
    last_emit = start

    i = start + 1
    while i < end:
        if not np.isfinite(values[i]):
            i += 1
            continue
        local = values[max(start, i - 1) : min(end, i + 2)]
        local = local[np.isfinite(local)]
        if not len(local):
            i += 1
            continue
        pitch = float(np.median(local))
        if abs(pitch - anchor_pitch) >= threshold and i - last_emit >= 2:
            anchors.append((i, pitch))
            anchor_pitch = pitch
            last_emit = i
        i += 1

    tail = values[max(start, end - 4) : end]
    tail = tail[np.isfinite(tail)]
    if len(tail):
        tail_pitch = float(np.median(tail))
        if abs(tail_pitch - anchors[-1][1]) >= threshold:
            anchors.append((max(start, end - 2), tail_pitch))
    return anchors


def _fine_pitch(midi_value: float, key: int) -> int:
    cents = (midi_value - key) * 100.0
    return int(round(120 + np.clip(cents, -120, 120)))


def analyze_audio_v2(path: str, cfg: AnalysisConfigV2):
    y, sr = _load(path, cfg.analysis_sr)
    midi, rms, voiced = _pitch_track(y, sr, cfg)

    hop_ms = cfg.hop_length * 1000.0 / sr
    voiced = _bridge(voiced, max(1, round(cfg.max_gap_ms / hop_ms)))
    min_frames = max(2, round(cfg.min_note_ms / hop_ms))
    sec_per_frame = cfg.hop_length / sr
    beat_per_sec = cfg.bpm / 60.0

    events: list[NoteEvent] = []
    debug_events = []

    for start, end in _region_bounds(voiced):
        if end - start < min_frames:
            continue

        anchors = _compress_curve(midi, start, end, cfg.slide_step_semitones)
        if not anchors:
            continue

        # Stabilize the first carrier around the first few voiced frames.
        carrier_pitch = anchors[0][1]
        carrier_key = int(np.clip(round(carrier_pitch), 0, 127))
        start_s = start * sec_per_frame
        end_s = end * sec_per_frame
        start_beat = _snap(start_s * beat_per_sec, cfg.grid_beat)
        end_beat = _snap(end_s * beat_per_sec, cfg.grid_beat)
        if end_beat <= start_beat:
            end_beat = start_beat + cfg.grid_beat

        carrier = NoteEvent(
            start_beat=start_beat,
            duration_beat=end_beat - start_beat,
            key=carrier_key,
            velocity=cfg.velocity,
            pan=64,
            fine_pitch=_fine_pitch(carrier_pitch, carrier_key),
            slide=False,
        )
        events.append(carrier)
        debug_events.append(
            {
                "type": "note",
                "start_beat": start_beat,
                "duration_beat": carrier.duration_beat,
                "key": carrier_key,
                "raw_midi": carrier_pitch,
            }
        )

        previous_pitch = carrier_pitch
        previous_frame = start
        for frame, pitch in anchors[1:]:
            delta = abs(pitch - previous_pitch)
            gap_ms = (frame - previous_frame) * hop_ms
            previous_frame = frame
            if delta < cfg.slide_min_delta:
                continue
            if gap_ms > cfg.slide_max_gap_ms and abs(round(pitch) - round(previous_pitch)) >= 1:
                # A large discontinuity is more likely a fresh syllable/note.
                slide_allowed = False
            else:
                slide_allowed = True

            beat = _snap(frame * sec_per_frame * beat_per_sec, cfg.grid_beat)
            key = int(np.clip(round(pitch), 0, 127))
            if slide_allowed:
                duration = cfg.grid_beat
                events.append(
                    NoteEvent(
                        start_beat=beat,
                        duration_beat=duration,
                        key=key,
                        velocity=cfg.velocity,
                        pan=64,
                        fine_pitch=_fine_pitch(pitch, key),
                        slide=True,
                    )
                )
                debug_events.append(
                    {
                        "type": "slide",
                        "start_beat": beat,
                        "duration_beat": duration,
                        "key": key,
                        "raw_midi": pitch,
                    }
                )
            else:
                # Preserve a strong discontinuity as another normal note.  This
                # avoids drawing a giant slide across separate vocal attacks.
                events.append(
                    NoteEvent(
                        start_beat=beat,
                        duration_beat=max(cfg.grid_beat, end_beat - beat),
                        key=key,
                        velocity=cfg.velocity,
                        pan=64,
                        fine_pitch=_fine_pitch(pitch, key),
                        slide=False,
                    )
                )
                debug_events.append(
                    {
                        "type": "note",
                        "start_beat": beat,
                        "duration_beat": max(cfg.grid_beat, end_beat - beat),
                        "key": key,
                        "raw_midi": pitch,
                        "reason": "pitch discontinuity",
                    }
                )
            previous_pitch = pitch

    # Deduplicate events that snap to the same grid location/key/type.
    unique = {}
    for event in events:
        token = (round(event.start_beat, 6), event.key, bool(event.slide))
        old = unique.get(token)
        if old is None or event.duration_beat > old.duration_beat:
            unique[token] = event
    events = sorted(unique.values(), key=lambda e: (e.start_beat, e.key, e.slide))

    return events, {
        "version": "0.2",
        "sample_rate": sr,
        "duration_s": len(y) / sr,
        "bpm": cfg.bpm,
        "pitch_offset": cfg.pitch_offset,
        "grid_beat": cfg.grid_beat,
        "event_count": len(events),
        "slide_count": sum(bool(e.slide) for e in events),
        "events": debug_events,
    }
