# FNF Audio → FLM Converter

Experimental **direct audio → FL Studio Mobile (.flm)** converter. MIDI is deliberately not used, because it loses FL Studio-specific semantics such as slide notes and makes the reference target less faithful.

## v0.2 scope

The converter currently targets **isolated monophonic FNF character vocals** and writes normal + slide events directly into FLM `EVN2`. v0.2 is the first build calibrated against a real audio + hand-authored FLM pair.

The FLM writer is derived from the existing `Vitkkk/Flp-converter-` project and its real FL Studio Mobile 4.10.17 DirectWave template. FLM note timing uses 128 units/beat and EVN2 v20.

### First real calibration findings

The first reference pair was tested at **167 BPM**. The reference FLM contains 211 EVN2 events across 6 clips, including 96 slide notes. Its shortest slide events are 1/16 beat (0.0625 beat, about 22.46 ms at 167 BPM).

The raw F0 tracker followed the main melody surprisingly closely, but this character's rendered vocal is detected two octaves below the piano-roll mapping in the reference FLM. For this pair, `--pitch-offset 24` is therefore required. A remaining isolated ±12-semitone YIN octave error showed that octave-continuity correction is an important next calibration target.

v0.2 changes:

- direct FLM output remains unchanged; no MIDI stage;
- faster YIN-based pitch tracking for long stems;
- configurable pitch offset;
- default 1/16-beat timing snap;
- pitch-curve simplification into **chains of FLM slide notes**, rather than only one slide between stable notes;
- shorter analysis frames/hops so ~22 ms slide events are representable;
- fixed reference-like note velocity by default to reduce irrelevant variation during calibration.

## Install

```bash
python -m pip install -e .
```

## Convert

The exact project BPM is required:

```bash
audio2flm vocals.wav --bpm 160 -o vocals_generated.flm --debug-json vocals_generated.json
```

For the first 167 BPM calibration vocal:

```bash
audio2flm "teste 1.mp3" --bpm 167 --pitch-offset 24 --grid 0.0625 -o teste1_generated.flm --debug-json teste1_generated.json
```

`--pitch-offset` is deliberately configurable rather than globally hard-coded because different FNF chromatics may use different piano-roll/sample mappings.

Supported input depends on libsndfile/FFmpeg available on the machine; WAV and FLAC remain the safest formats, but MP3 can be read when FFmpeg support is available.

## Calibration workflow

For each new pass, keep/provide:

1. isolated vocal audio;
2. generated FLM;
3. original hand-authored/correct FLM containing the same vocal notes;
4. debug JSON;
5. exact BPM.

The reference is read in the FLM domain itself. MIDI is never used as ground truth.

## Current limitations

- Main detector is still monophonic; simultaneous octave-doubled events in a reference FLM are not reconstructed yet.
- BPM must be supplied manually.
- Pitch-offset calibration is currently manual.
- Isolated octave-tracking mistakes can still occur and will be addressed with temporal octave unwrapping.
- Full pattern/playlist reconstruction is not yet the goal; the current calibration target is note/slide accuracy.
- Fine pitch is still approximate pending more real reference pairs.

## Architecture

`audio -> F0/RMS analysis -> vocal regions -> pitch-curve anchors -> normal/slide NoteEvent list -> native FLM EVN2 writer`

No MIDI stage exists anywhere in the conversion path.
