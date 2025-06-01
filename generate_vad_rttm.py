import argparse
import os
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from pyannote.core import Annotation, Segment

from utils import MyNotebook
from vad import apply_vad_pipeline


def _trim_silence(annotation: Annotation, window_ms: int = 200, copy: bool = False) -> Annotation:
    if copy:
        annotation = annotation.copy()
    duration = annotation.get_timeline().duration()
    for segment, track, label in list(annotation.itertracks(yield_label=True)):
        del annotation[segment, track]
        start = max(0, segment.start - window_ms / 1000.0)
        end = min(duration, segment.end + window_ms / 1000.0)
        new_segment = Segment(start, end)
        annotation[new_segment, track] = label
    return annotation


def process_input_file(
    audio_input_file: str | os.PathLike,
    label: str,
    output_dir: str | os.PathLike,
    draw: bool = False,
    trim_silence_window_ms: int = 0,
) -> Annotation:
    logger.info(f"Processing audio file: {audio_input_file} (label: {label})")
    logger.info("Applying VAD pipeline...")
    annotation = apply_vad_pipeline(audio_input_file)
    annotation.rename_labels({"SPEECH": label}, copy=False)
    annotation.uri = label
    if trim_silence_window_ms > 0:
        annotation = _trim_silence(annotation, window_ms=trim_silence_window_ms)
    if draw:
        notebook = MyNotebook()
        notebook.plot_annotation(annotation)
        plots_output_dir = Path(output_dir) / "plots"
        plots_output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_output_dir / f"{label}-annotation.png")
    return annotation


def main(
    audio_file: str | os.PathLike,
    output_dir: str | os.PathLike,
    draw: bool = False,
    trim_silence_window_ms: int = 0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = Path(audio_file).stem
    annotation = process_input_file(
        audio_file, label, output_dir, draw=draw, trim_silence_window_ms=trim_silence_window_ms
    )
    output_rttm = output_dir / f"{label}.rttm"
    Path(output_rttm).write_text(annotation.to_rttm())
    logger.success(f"Done! VAD segments saved to {output_rttm}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate speaker diarization ground truth using VAD.")
    parser.add_argument("--audio-file", "-a", type=str, help="Path to the audio file.", required=True)
    parser.add_argument("--output", "-o", type=str, help="Path to save the outputs", default="output")
    parser.add_argument("--draw", action="store_true", help="Whether to draw the annotation plot.")
    parser.add_argument(
        "--trim_silence_window_ms",
        "-t",
        type=int,
        default=0,
        help="Window size in milliseconds to trim silence around segments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info("Starting ASR VAD overlap detection...")
    logger.info(f"Arguments: {args}")
    main(
        audio_file=args.audio_file,
        output_dir=args.output,
        draw=args.draw,
        trim_silence_window_ms=args.trim_silence_window_ms,
    )
