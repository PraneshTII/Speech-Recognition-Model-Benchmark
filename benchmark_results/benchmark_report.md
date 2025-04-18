# Speech Recognition Model Benchmark Report

Generated on: 2025-04-18 18:24:50

## Models Benchmarked

- WhisperCPP: 10 models
- Faster-Whisper: 4 models
- Vosk: 3 models

## Audio Files

- audio_2m.mp3 (119.77 seconds)
- audio_3m.mp3 (230.01 seconds)
- audio_1m.mp3 (56.71 seconds)

## Performance Summary

- **Fastest Model Overall**: Faster-Whisper tiny.en (RTF: 0.0770)
- **Most Accurate Model**: WhisperCPP ggml-base.en.bin (WER: 0.0454)

## Detailed Results

### Results for audio_2m.mp3

| Model | RTF | Processing Time | WER | | --- | --- | --- | --- | 
| Faster-Whisper tiny.en | 0.0770 | 9.22s | 0.1338 | 
| Faster-Whisper base.en | 0.1178 | 14.10s | 0.0855 | 
| WhisperCPP ggml-tiny-q8_0.bin | 0.1354 | 16.22s | 0.1375 | 
| Vosk vosk-model-small-en-us-0.15.zip | 0.1509 | 18.07s | 0.3978 | 
| WhisperCPP ggml-tiny-q5_1.bin | 0.1643 | 19.68s | 0.1561 | 
| WhisperCPP ggml-base.en-q8_0.bin | 0.2466 | 29.54s | 0.1190 | 
| WhisperCPP ggml-tiny.bin | 0.2715 | 32.52s | 0.1338 | 
| WhisperCPP ggml-base-q5_1.bin | 0.3021 | 36.18s | 0.1190 |
| WhisperCPP ggml-base.en-q5_1.bin | 0.3038 | 36.39s | 0.0818 | 
| Faster-Whisper small.en | 0.3332 | 39.91s | 0.0595 | 
| Vosk vosk-model-en-us-0.22.zip | 0.5243 | 62.80s | 0.3271 | 
| WhisperCPP ggml-base.en.bin | 0.5487 | 65.72s | 0.1264 | 
| Vosk vosk-model-en-us-0.22-lgraph.zip | 0.5839 | 69.93s | 0.3532 |
| WhisperCPP ggml-small-q5_1.bin | 0.9591 | 114.87s | 0.0967 | 
| Faster-Whisper medium.en | 0.9832 | 117.76s | 0.0967 | 
| WhisperCPP ggml-medium-q8_0.bin | 2.0502 | 245.55s | 0.0706 |
| WhisperCPP ggml-medium-q5_0.bin | 2.3596 | 282.61s | 0.1004 | 


### Results for audio_3m.mp3

| Model | RTF | Processing Time | WER | | --- | --- | --- | --- | 
| Faster-Whisper tiny.en | 0.0877 | 20.18s | 0.0859 | 
| Faster-Whisper base.en | 0.1339 | 30.79s | 0.0583 | 
| WhisperCPP ggml-tiny-q8_0.bin | 0.1443 | 33.18s | 0.0940 | 
| Vosk vosk-model-small-en-us-0.15.zip | 0.1513 | 34.80s | 0.3647 | 
| WhisperCPP ggml-tiny-q5_1.bin | 0.1703 | 39.17s | 0.0843 | 
| WhisperCPP ggml-base.en-q8_0.bin | 0.2541 | 58.44s | 0.0600 |
| WhisperCPP ggml-tiny.bin | 0.2961 | 68.11s | 0.1037 | 
| WhisperCPP ggml-base.en-q5_1.bin | 0.3057 | 70.32s | 0.0486 |
| WhisperCPP ggml-base-q5_1.bin | 0.3142 | 72.27s | 0.0875 | 
| Faster-Whisper small.en | 0.3399 | 78.19s | 0.2334 |
| WhisperCPP ggml-base.en.bin | 0.5587 | 128.50s | 0.0454 | 
| Vosk vosk-model-en-us-0.22.zip | 0.5587 | 128.52s | 0.3015 |
| Vosk vosk-model-en-us-0.22-lgraph.zip | 0.6683 | 153.71s | 0.3241 | 
| WhisperCPP ggml-small-q5_1.bin | 0.8746 | 201.16s | 0.0891 |
| Faster-Whisper medium.en | 0.9573 | 220.18s | 0.0681 | 
| WhisperCPP ggml-medium-q8_0.bin | 1.9102 | 439.37s | 0.0665 |
| WhisperCPP ggml-medium-q5_0.bin | 2.2689 | 521.87s | 0.0648 | 


### Results for audio_1m.mp3

| Model | RTF | Processing Time | WER | | --- | --- | --- | --- | 
| Faster-Whisper tiny.en | 0.0829 | 4.70s | 0.1301 |
| WhisperCPP ggml-tiny-q8_0.bin | 0.1281 | 7.27s | 0.2055 | 
| Faster-Whisper base.en | 0.1467 | 8.32s | 0.1438 |
| WhisperCPP ggml-tiny-q5_1.bin | 0.1979 | 11.22s | 0.2123 |
| Vosk vosk-model-small-en-us-0.15.zip | 0.2057 | 11.67s | 0.5685 | 
| WhisperCPP ggml-tiny.bin | 0.2571 | 14.58s | 0.1849 | 
| WhisperCPP ggml-base.en-q5_1.bin | 0.2906 | 16.48s | 0.1438 | 
| WhisperCPP ggml-base-q5_1.bin | 0.2912 | 16.51s | 0.1438 | 
| WhisperCPP ggml-base.en-q8_0.bin | 0.3035 | 17.21s | 0.1507 |
| WhisperCPP ggml-base.en.bin | 0.6563 | 37.22s | 0.1781 | 
| Vosk vosk-model-en-us-0.22-lgraph.zip | 0.7587 | 43.02s | 0.5205 | 
| Vosk vosk-model-en-us-0.22.zip | 0.7744 | 43.92s | 0.4795 | 
| WhisperCPP ggml-small-q5_1.bin | 0.8657 | 49.09s | 0.1507 | 
| Faster-Whisper medium.en | 1.0868 | 61.63s | 0.1301 | 
| Faster-Whisper small.en | 1.5035 | 85.27s | 0.0959 |
| WhisperCPP ggml-medium-q8_0.bin | 2.4347 | 138.08s | 0.1301 |
| WhisperCPP ggml-medium-q5_0.bin | 2.9340 | 166.39s | 0.1507 | 

## Visualizations

Charts are available in the `benchmark_results/charts/` directory:

- [summary_audio_3m.png](charts/summary_audio_3m.png)
- [summary_audio_2m.png](charts/summary_audio_2m.png)
- [latency_comparison.png](charts/latency_comparison.png)
- [summary_audio_1m.png](charts/summary_audio_1m.png)
- [wer_comparison.png](charts/wer_comparison.png)
- [rtf_comparison.png](charts/rtf_comparison.png)
