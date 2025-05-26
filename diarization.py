import os
import warnings
from typing import Callable

import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from pyannote.core import Annotation

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    from pyannote.audio import Pipeline

load_dotenv()
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable with your Hugging Face token."


def apply_diarization_pipeline(
    audio_file: str | os.PathLike, duration_seconds: int | None = None, hook: Callable | None = None
) -> Annotation:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    logger.debug(f"Processing {audio_file} on {device}")
    waveform, sample_rate = torchaudio.load(audio_file)
    if duration_seconds is not None:
        logger.debug(f"Trimming waveform to {duration_seconds} seconds")
        max_samples = sample_rate * duration_seconds
        waveform = waveform[:, :max_samples]
    input_ = {"waveform": waveform, "sample_rate": sample_rate}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*MPEG_LAYER_III.*")
        output = pipeline(input_, hook=hook)
    return output


if __name__ == "__main__":
    audio_file = "data/mixed.mp3"
    vad_result = apply_diarization_pipeline(audio_file, duration_seconds=60)
    segments = vad_result.get_timeline().support()
    for i, segment in enumerate(segments):
        print(f"Segment {i}: start={segment.start:.1f}s, end={segment.end:.1f}s")
