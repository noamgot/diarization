import argparse
import os
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from pyannote.core import Annotation, Segment, Timeline

from utils import MyNotebook, json_to_annotation, timeline_to_single_label_annotation
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


def get_asr_vad_overlap(
    name: str,
    audio_input_file: str | os.PathLike,
    asr_json_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    draw: bool = False,
    trim_silence_window_ms: int = 0,
) -> Timeline:
    logger.info(f"Processing audio file: {audio_input_file} and ASR JSON: {asr_json_path}")

    logger.info("Applying VAD pipeline...")
    vad_annotation = apply_vad_pipeline(audio_input_file)
    vad_annotation.rename_labels({"SPEECH": "VAD"}, copy=False)
    if trim_silence_window_ms > 0:
        vad_annotation = _trim_silence(vad_annotation, window_ms=trim_silence_window_ms)
    logger.info("Loading ASR segments from JSON...")
    asr_segments = json_to_annotation(asr_json_path, label="ASR")

    combined_annotation = Annotation()
    combined_annotation.update(vad_annotation)
    combined_annotation.update(asr_segments)

    overlap_tl = combined_annotation.get_overlap()
    for segment in overlap_tl:
        combined_annotation[segment] = "OVERLAP"
    if draw:
        notebook = MyNotebook()
        notebook.plot_annotation(combined_annotation, separate_by="labels")
        plots_output_dir = Path(output_dir) / "plots"
        plots_output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_output_dir / f"{name}-annotation.png")
    return overlap_tl


def main(
    audio_file: str | os.PathLike,
    asr_json: str | os.PathLike,
    output_dir: str | os.PathLike,
    draw: bool = False,
    trim_silence_window_ms: int = 0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(audio_file).stem
    overlap_timeline = get_asr_vad_overlap(
        name, audio_file, output_dir, asr_json, draw=draw, trim_silence_window_ms=trim_silence_window_ms
    )
    overlap_annotation = timeline_to_single_label_annotation(overlap_timeline, label=name)
    overlap_annotation.uri = name
    output_rttm = output_dir / f"{name}.rttm"
    Path(output_rttm).write_text(overlap_annotation.to_rttm())
    logger.success(f"Done! Overlap segments saved to {output_rttm}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process audio and ASR JSON to find overlap segments.")
    parser.add_argument("--audio-file", "-a", type=str, help="Path to the audio file.", required=True)
    parser.add_argument("--asr-json", "-j", type=str, help="Path to the ASR JSON file.", required=True)
    parser.add_argument("--output", "-o", type=str, help="Path to save the outputs", default="output")
    parser.add_argument("--draw", action="store_true", help="Whether to draw the annotation plot.")
    parser.add_argument(
        "--trim_silence_window_ms",
        type=int,
        default=0,
        help="Window size in milliseconds to trim silence around segments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting ASR VAD overlap detection...")
    logger.info(f"Arguments: {args}")
    main(
        audio_file=args.audio_file,
        asr_json=args.asr_json,
        output_dir=args.output,
        draw=args.draw,
        trim_silence_window_ms=args.trim_silence_window_ms,
    )
