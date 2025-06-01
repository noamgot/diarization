import argparse
import os
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from pyannote.core import Annotation, Segment
from tqdm import tqdm

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
    output_dir: str | os.PathLike,
    draw: bool = False,
    trim_silence_window_ms: int = 0,
) -> Annotation:
    label = Path(audio_input_file).stem
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


def main(input_files: list[str], output_dir: str | os.PathLike):
    logger.info(f"Starting VAD processing for {len(input_files)} input files.")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mixed_annotation = Annotation(uri="mixed")
    for audio_file in tqdm(input_files):
        annotation = process_input_file(audio_file, output_dir=output_dir)
        Path(output_dir / f"{annotation.uri}.rttm").write_text(annotation.to_rttm())
        mixed_annotation.update(annotation)
    Path(output_dir / "mixed.rttm").write_text(mixed_annotation.to_rttm())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Generate ground-truth diarization data:
            1. Apply VAD to each of the input audio files.
            2. Mix the results into a single RTTM file.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-files",
        "-i",
        help="Path to an input audio files. For more than one file, use space-separated paths.",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-dir", "-o", type=str, help="Directory to save output RTTM files.", required=True)

    args = parser.parse_args()
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    main(args.input_files, args.output_dir)
