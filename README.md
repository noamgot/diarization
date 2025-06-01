# Diarization Project

## Development Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To set up the development environment:

1. **Install uv** (if not already installed) - follow the instructions on the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).


2. **Run everything in a virtual environment:**

   ```sh
   uv run <command>
   ```

    In the first run, uv will create a virtual environment and install all dependencies specified in `pyproject.toml`.
---

## Running `generate_ground_truth.py`

This script applies Voice Activity Detection (VAD) to input audio files and generates ground-truth diarization RTTM files.

**Usage:**

```sh
uv run generate_ground_truth.py --input-files <audio_file1> <audio_file2> ... --output-dir <output_directory>
```

- `<audio_file1> <audio_file2> ...` — One or more paths to input audio files (space-separated).
- `<output_directory>` — Directory where the RTTM files will be saved.

**Example:**

```sh
uv run generate_ground_truth.py --input-files data/ch1-netfrix.mp3 data/ch2-nitzan-hadar.mp3 --output-dir output/netfrix
```

This will process the input files and save the resulting RTTM files in the specified output directory.
