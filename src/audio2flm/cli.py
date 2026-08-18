from __future__ import annotations

import argparse
import json
import pathlib

from .analyzer_v2 import AnalysisConfigV2, analyze_audio_v2
from .flm import write_flm


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="audio2flm",
        description=(
            "Converte vocal monofônica diretamente para FL Studio Mobile .flm, "
            "incluindo cadeias de slide notes."
        ),
    )
    parser.add_argument("audio")
    parser.add_argument("-o", "--output", help="Arquivo .flm de saída")
    parser.add_argument("--bpm", type=float, required=True, help="BPM exato da música/projeto")
    parser.add_argument("--name", default="Audio to FLM v0.2", help="Nome do projeto no FLM")
    parser.add_argument("--channel", default="Vocal", help="Nome do canal DirectWave")
    parser.add_argument("--fmin", default="C2")
    parser.add_argument("--fmax", default="C7")
    parser.add_argument(
        "--pitch-offset",
        type=float,
        default=0.0,
        help="Transposição aplicada ao F0 detectado, em semitons (ex.: 24)",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=1.0 / 16.0,
        help="Snap em beats; padrão 0.0625 (1/16 de beat)",
    )
    parser.add_argument("--debug-json", help="Salva detecções para calibração posterior")
    args = parser.parse_args(argv)

    source = pathlib.Path(args.audio)
    output = pathlib.Path(args.output) if args.output else source.with_suffix(".flm")

    config = AnalysisConfigV2(
        bpm=args.bpm,
        fmin=args.fmin,
        fmax=args.fmax,
        pitch_offset=args.pitch_offset,
        grid_beat=args.grid,
    )
    events, debug = analyze_audio_v2(str(source), config)
    output.write_bytes(
        write_flm(
            events,
            project_name=args.name,
            channel_name=args.channel,
            bpm=args.bpm,
        )
    )

    slide_count = sum(bool(event.slide) for event in events)
    debug.update(
        {
            "output": str(output),
            "notes": len(events) - slide_count,
            "slides": slide_count,
        }
    )
    if args.debug_json:
        pathlib.Path(args.debug_json).write_text(
            json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "version": "0.2",
                "output": str(output),
                "notes": len(events) - slide_count,
                "slides": slide_count,
                "pitch_offset": args.pitch_offset,
                "grid": args.grid,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
