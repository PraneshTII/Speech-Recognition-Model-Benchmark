# Speech Recognition Benchmark Tool

A comprehensive benchmarking script for comparing speech recognition models across different frameworks (WhisperCPP, Faster-Whisper, and Vosk).

## Overview

This tool allows you to benchmark and compare the performance of various speech recognition models on your audio files. It measures:

- **Real-Time Factor (RTF)** - How fast the model processes audio compared to audio duration
- **Word Error Rate (WER)** - Accuracy of transcription (when reference transcriptions are provided)


## Directory Structure

The script expects the following directory structure:

```
.
├── audio_transcription_script.py
├── models
│   ├── faster_whisper_models
│   │   ├── models--Systran--faster-whisper-base.en
│   │   ├── models--Systran--faster-whisper-medium.en
│   │   ├── models--Systran--faster-whisper-small.en
│   │   └── models--Systran--faster-whisper-tiny.en
│   ├── vosk_models
│   │   └── *.zip
│   └── whispercpp_models
│       └── *.bin
├── binary
│   └── build
│       └── bin
│           └── whisper-cli
└── sample_audio
    ├── audio_1m.mp3
    ├── audio_2m.mp3
    └── audio_3m.mp3
├── test_audio_py (virtual environment)
```

### Model Sources

- **WhisperCPP models**: Download from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
- **Vosk models**: Download from [Alphacephei](https://alphacephei.com/vosk/models)
- **Faster-Whisper models**: Downloaded automatically upon first use

## Installation

### Prerequisites

- Python 3.6+
- ffmpeg (for audio duration measurement)
- Git

### Setting Up Virtual Environment

Create a virtual environment for the benchmark tools:

```bash
python3 -m venv test_audio_py
source test_audio_py/bin/activate
```

### Installing and Setting Up Vosk

```bash
# Install Vosk
pip install vosk

# Test Vosk installation
vosk-transcriber -i path/to/test.wav -o test_transcribe.txt
```

Download Vosk models from: https://alphacephei.com/vosk/models
Place downloaded models in `models/vosk_models/` directory.

### Installing and Setting Up Faster-Whisper

```bash
# Install Faster-Whisper
pip install faster-whisper
```

Faster-Whisper models will be downloaded automatically when first used. For example:
```python
from faster_whisper import WhisperModel
model = WhisperModel(model_size, device="cpu", compute_type="int8")
```

### Installing and Setting Up WhisperCPP

```bash
# Clone the repository
git clone https://github.com/ggml-org/whisper.cpp.git

# Navigate to the directory
cd whisper.cpp

# Install required dependencies (Ubuntu/Debian)
sudo apt-get install libomp-dev

# Build WhisperCPP
cmake -B build
cmake --build build --config Release

# Test the installation
./build/bin/whisper-cli -f samples/jfk.wav
```

Download pre-trained WhisperCPP models from: https://huggingface.co/ggerganov/whisper.cpp/tree/main
Place downloaded models in `models/whispercpp_models/` directory.

### Required Python Packages for the Benchmark Script

```bash
pip install numpy pandas matplotlib psutil jiwer
```

## Usage

### List Available Models

```bash
python audio_transcription_script.py --list_models
```

### Create Reference Transcriptions

To calculate WER (Word Error Rate), you need reference transcriptions:

```bash
python audio_transcription_script.py --create_references
```

This creates template files in `reference_transcriptions/` which you should edit with accurate transcriptions.

### Run Benchmarks

Run all available models:

```bash
python audio_transcription_script.py --all
```

Run specific model types:

```bash
python audio_transcription_script.py --whispercpp --faster_whisper
```

Run specific models:

```bash
python audio_transcription_script.py --specific_models models/whispercpp_models/ggml-base.en.bin tiny.en
```

### Command Line Options

```
--audio_dir DIR       Directory containing audio files (default: ./sample_audio)
--ref_dir DIR         Directory containing reference transcriptions (default: ./reference_transcriptions)
--output_dir DIR      Directory to store results (default: ./benchmark_results)
--create_references   Create template reference transcription files
--whispercpp          Run WhisperCPP benchmarks
--faster_whisper      Run Faster-Whisper benchmarks
--vosk                Run Vosk benchmarks
--all                 Run all available models
--list_models         List available models
--specific_models     Run specific models (provide full paths)
```

## Output

The benchmark script generates:

1. Transcriptions for each model-audio combination
2. CSV file with benchmark metrics
3. JSON file with detailed results
4. Charts comparing model performance
5. A comprehensive markdown report

## Understanding RTF (Real-Time Factor)

RTF represents the ratio between processing time and audio duration:
- RTF < 1.0: Model is faster than real-time (lower is better)
- RTF = 1.0: Model processes at exactly real-time
- RTF > 1.0: Model is slower than real-time

For example, an RTF of 0.5 means the model can process audio twice as fast as real-time.

## Understanding WER (Word Error Rate)

WER measures the accuracy of the transcription by comparing it to a reference:

```
WER = (Substitutions + Deletions + Insertions) / Number of Words in Reference
```

- WER = 0.0: Perfect transcription (lower is better)
- WER = 1.0: Complete mismatch with reference
- WER > 1.0: Possible when there are many insertions

For meaningful WER calculation, you must provide accurate reference transcriptions. The script will help you create these with the `--create_references` option, but you'll need to manually verify and correct them for your audio files.

## Results 

Results are available in the `benchmark_results/` directory:

Summarized Report: - [benchmark_report.md](/benchmark_results/benchmark_report.md)

- [summary_audio_3m.png](charts/summary_audio_3m.png)
- [summary_audio_2m.png](charts/summary_audio_2m.png)
- [latency_comparison.png](charts/latency_comparison.png)
- [summary_audio_1m.png](charts/summary_audio_1m.png)
- [wer_comparison.png](charts/wer_comparison.png)
- [rtf_comparison.png](charts/rtf_comparison.png)


Note that WER does not account for semantic correctness - it's a strict word-level comparison. Two transcriptions with the same meaning but different wording will still show differences in WER.

