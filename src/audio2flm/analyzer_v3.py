from __future__ import annotations
from dataclasses import dataclass
import librosa
import numpy as np
from scipy.ndimage import median_filter
from .flm import NoteEvent

@dataclass
class AnalysisConfigV3:
    bpm: float = 120.0
    fmin: str = 'C2'
    fmax: str = 'C7'
    analysis_sr: int = 22050
    frame_length: int = 1024
    hop_length: int = 128
    pitch_offset: float = 0.0
    grid_beat: float = 1/16
    min_note_ms: float = 42.0
    max_gap_ms: float = 24.0
    abrupt_change_semitones: float = 1.15
    stable_change_semitones: float = 0.68
    stable_hold_ms: float = 42.0
    onset_delta: float = 0.18
    slide_min_delta: float = 0.85
    slide_min_ms: float = 32.0
    slide_max_ms: float = 240.0
    slide_anchor_step: float = 1.35
    octave_jump_threshold: float = 7.0
    velocity: int = 102


def _snap(v, g):
    return v if g <= 0 else round(v / g) * g


def _load(path, sr):
    y, actual_sr = librosa.load(path, sr=sr, mono=True)
    if not len(y):
        raise ValueError('Áudio vazio.')
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), int(actual_sr)


def _bridge(mask, max_frames):
    out = mask.copy(); i = 0
    while i < len(out):
        if out[i]:
            i += 1; continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and (j-i) <= max_frames:
            out[i:j] = True
        i = j
    return out


def _octave_correct(midi, voiced, threshold):
    out = midi.copy()
    history = []
    for i in range(len(out)):
        if not voiced[i] or not np.isfinite(out[i]):
            continue
        raw = float(out[i])
        if history:
            ref = float(np.median(history[-7:]))
            if abs(raw-ref) >= threshold:
                candidates = [raw + k for k in (-24,-12,0,12,24)]
                best = min(candidates, key=lambda x: abs(x-ref))
                if abs(best-ref) + 2.25 < abs(raw-ref):
                    raw = best
                    out[i] = raw
        history.append(raw)
    return out


def _track(y, sr, cfg):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz(cfg.fmin), fmax=librosa.note_to_hz(cfg.fmax), sr=sr, frame_length=cfg.frame_length, hop_length=cfg.hop_length)
    midi = librosa.hz_to_midi(f0) + cfg.pitch_offset
    rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
    n = min(len(midi), len(rms), len(onset)); midi=midi[:n]; rms=rms[:n]; onset=onset[:n]
    finite = np.isfinite(midi)
    nz = rms[rms > 1e-7]
    gate = max(0.004, float(np.percentile(nz, 24))*1.15) if len(nz) else 0.004
    voiced = finite & (rms >= gate)
    midi = _octave_correct(midi, voiced, cfg.octave_jump_threshold)
    temp = midi.copy(); idx=np.arange(n)
    if finite.any():
        temp[~finite] = np.interp(idx[~finite], idx[finite], temp[finite])
        temp = median_filter(temp, size=7, mode='nearest')
        midi[finite] = temp[finite]
    onset = onset / max(1e-9, float(np.max(onset)))
    return midi, rms, onset, voiced


def _regions(mask):
    i=0
    while i < len(mask):
        if not mask[i]: i+=1; continue
        j=i+1
        while j < len(mask) and mask[j]: j+=1
        yield i,j
        i=j


def _fine_pitch(m, key):
    return int(round(120 + np.clip((m-key)*100, -120, 120)))


def _stable_median(midi, a, b):
    x=midi[max(0,a):min(len(midi),b)]
    x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _find_boundaries(midi, onset, start, end, cfg, hop_ms):
    hold = max(3, round(cfg.stable_hold_ms/hop_ms)); candidates = {start, end}; i = start + hold
    while i < end-hold:
        left = _stable_median(midi, i-hold, i); right = _stable_median(midi, i, i+hold)
        if not np.isfinite(left) or not np.isfinite(right):
            i += 1; continue
        delta = abs(right-left)
        raw_jump = abs(float(midi[i]) - float(midi[i-1])) if np.isfinite(midi[i]) and np.isfinite(midi[i-1]) else 0
        strong_onset = float(np.max(onset[max(start,i-2):min(end,i+3)])) >= cfg.onset_delta
        if delta >= cfg.stable_change_semitones and (strong_onset or raw_jump >= cfg.abrupt_change_semitones):
            candidates.add(i); i += hold; continue
        i += 1
    return sorted(candidates)


def _is_slide(midi, onset, left_end, right_start, region_start, region_end, cfg, hop_ms):
    span = max(4, round(cfg.slide_max_ms/hop_ms)); a=max(region_start, left_end-span//2); b=min(region_end, right_start+span//2)
    if b-a < 5: return False, []
    x=midi[a:b]; good=np.isfinite(x)
    if np.sum(good) < 5: return False, []
    vals=x[good]; start_pitch=float(np.median(vals[:min(4,len(vals))])); end_pitch=float(np.median(vals[-min(4,len(vals)):]))
    total=end_pitch-start_pitch
    if abs(total) < cfg.slide_min_delta: return False, []
    onset_peak=float(np.max(onset[max(region_start,right_start-2):min(region_end,right_start+3)]))
    if onset_peak >= cfg.onset_delta*1.35: return False, []
    lo,hi=sorted((start_pitch,end_pitch)); intermediate=np.sum((vals > lo+0.18) & (vals < hi-0.18)); duration_ms=(b-a)*hop_ms
    if duration_ms < cfg.slide_min_ms or intermediate < 3: return False, []
    dif=np.diff(vals); directional=np.mean(dif >= -0.16) if total > 0 else np.mean(dif <= 0.16)
    if directional < 0.68: return False, []
    if len(dif) and np.max(np.abs(dif)) > max(3.5, abs(total)*0.82): return False, []
    anchors=[]; last=start_pitch
    for idx in range(2, len(vals)-1):
        p=float(np.median(vals[max(0,idx-1):min(len(vals),idx+2)]))
        if abs(p-last) >= cfg.slide_anchor_step:
            frac=idx/max(1,len(vals)-1); frame=int(round(a + frac*(b-a-1))); anchors.append((frame,p)); last=p
    if not anchors or abs(anchors[-1][1]-end_pitch) >= 0.6: anchors.append((right_start,end_pitch))
    return True, anchors[-3:]


def analyze_audio_v3(path, cfg: AnalysisConfigV3):
    y,sr=_load(path,cfg.analysis_sr); midi,rms,onset,voiced=_track(y,sr,cfg)
    hop_ms=cfg.hop_length*1000.0/sr; voiced=_bridge(voiced,max(1,round(cfg.max_gap_ms/hop_ms)))
    sec_per_frame=cfg.hop_length/sr; beat_per_sec=cfg.bpm/60.0; min_frames=max(2,round(cfg.min_note_ms/hop_ms)); events=[]; debug=[]
    for rs,re in _regions(voiced):
        if re-rs < min_frames: continue
        bounds=_find_boundaries(midi,onset,rs,re,cfg,hop_ms); cleaned=[bounds[0]]
        for b in bounds[1:-1]:
            if b-cleaned[-1] >= min_frames and re-b >= min_frames: cleaned.append(b)
        cleaned.append(re); bounds=cleaned; notes=[]
        for a,b in zip(bounds[:-1],bounds[1:]):
            if b-a < min_frames: continue
            p=_stable_median(midi,a,b)
            if np.isfinite(p): notes.append((a,b,p))
        if not notes: continue
        slide_map={}
        for i in range(len(notes)-1):
            a,b,p=notes[i]; c,d,q=notes[i+1]; ok,anchors=_is_slide(midi,onset,b,c,rs,re,cfg,hop_ms)
            if ok: slide_map[i]=anchors
        for i,(a,b,p) in enumerate(notes):
            start_beat=_snap(a*sec_per_frame*beat_per_sec,cfg.grid_beat); end_beat=_snap(b*sec_per_frame*beat_per_sec,cfg.grid_beat)
            if end_beat <= start_beat: end_beat=start_beat+cfg.grid_beat
            key=int(np.clip(round(p),0,127)); events.append(NoteEvent(start_beat,end_beat-start_beat,key,cfg.velocity,64,_fine_pitch(p,key),False))
            debug.append({'type':'note','start_beat':start_beat,'duration_beat':end_beat-start_beat,'key':key,'raw_midi':p})
            if i in slide_map:
                for frame,sp in slide_map[i]:
                    beat=_snap(frame*sec_per_frame*beat_per_sec,cfg.grid_beat); sk=int(np.clip(round(sp),0,127))
                    if sk == key and abs(beat-start_beat) < cfg.grid_beat*0.5: continue
                    events.append(NoteEvent(beat,cfg.grid_beat,sk,cfg.velocity,64,_fine_pitch(sp,sk),True)); debug.append({'type':'slide','start_beat':beat,'duration_beat':cfg.grid_beat,'key':sk,'raw_midi':sp})
    events=sorted(events,key=lambda e:(e.start_beat,e.slide,e.key)); kept=[]
    for e in events:
        duplicate=False
        for k in kept[-8:]:
            if abs(k.start_beat-e.start_beat) <= cfg.grid_beat*0.25:
                if k.key==e.key and k.slide==e.slide: duplicate=True; break
                if abs(k.key-e.key) in (12,24) and not e.slide and not k.slide: duplicate=True; break
        if not duplicate: kept.append(e)
    events=kept
    return events, {'version':'0.3','sample_rate':sr,'duration_s':len(y)/sr,'bpm':cfg.bpm,'pitch_offset':cfg.pitch_offset,'grid_beat':cfg.grid_beat,'event_count':len(events),'slide_count':sum(bool(e.slide) for e in events),'events':debug}
