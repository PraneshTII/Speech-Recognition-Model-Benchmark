#!/usr/bin/env python3

import os
import time
import json
import argparse
import subprocess
import psutil
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import shutil

# Try to import jiwer for WER calculation if available
try:
    from jiwer import wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("jiwer not found. WER calculation will be disabled.")

class ModelBenchmark:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.reference_texts = {}
        
    def load_references(self, reference_dir):
        """Load reference transcriptions if available"""
        if not os.path.exists(reference_dir):
            print(f"Reference directory {reference_dir} not found.")
            return
            
        for filename in os.listdir(reference_dir):
            if filename.endswith("_reference.txt"):
                audio_name = filename.replace("_reference.txt", "")
                with open(os.path.join(reference_dir, filename), 'r') as f:
                    content = f.read()
                    # Skip lines starting with # (comments)
                    lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
                    self.reference_texts[audio_name] = '\n'.join(lines).strip()

    def measure_whispercpp(self, model_path, audio_file):
        """Benchmark WhisperCPP"""
        model_name = os.path.basename(model_path)
        audio_name = os.path.basename(audio_file)
        print(f"Benchmarking WhisperCPP with {model_name} on {audio_name}")
        
        # Prepare output file base name (without extension)
        output_base = os.path.join(self.output_dir, f"whispercpp_{model_name}_{audio_name}")
        
        # Start monitoring resources
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=0.1)  # Reset CPU measurement
        
        # Time model loading and processing
        start_time = time.time()
        
        # Run whispercpp-cli with appropriate parameters
        # Based on the error message, we need to use -otxt and -of instead of -o
        cmd = [
            "./binary/build/bin/whisper-cli", 
            "-m", model_path, 
            "-f", audio_file, 
            "-otxt",  # Output as text
            "-of", output_base  # Output file path without extension
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        cpu_percent_end = psutil.cpu_percent(interval=None)
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        
        # Get audio duration
        audio_duration = self._get_audio_duration(audio_file)
        rtf = total_time / audio_duration if audio_duration else None
        
        # Extract transcription from output file (with .txt extension)
        output_file = f"{output_base}.txt"
        transcription = ""
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                transcription = f.read()
        
        # Calculate WER if reference text is available
        error_rate = None
        audio_base = os.path.splitext(audio_name)[0]
        if HAS_JIWER and audio_base in self.reference_texts:
            error_rate = wer(self.reference_texts[audio_base], transcription)
        
                
        # Store results
        result_key = f"whispercpp_{model_name}_{audio_name}"
        self.results[result_key] = {
            "model": f"WhisperCPP {model_name}",
            "audio": audio_name,
            "total_time": total_time,
            "rtf": rtf,
            "wer": error_rate,
            #"memory_usage_mb": end_memory - start_memory,
            #"cpu_usage": cpu_percent_end,
            #"model_load_time": model_load_time,
            "transcription": transcription
        }
        
        return self.results[result_key]
    
    def measure_faster_whisper(self, model_name, audio_file):
        """Benchmark faster-whisper"""
        audio_name = os.path.basename(audio_file)
        print(f"Benchmarking Faster-Whisper {model_name} on {audio_name}")
        
        # Prepare output file
        output_file = os.path.join(self.output_dir, f"faster_whisper_{model_name}_{audio_name}.txt")
        
        # Import faster_whisper here to measure its resource usage properly
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("faster_whisper not installed. Skipping this benchmark.")
            return None
        
        # Force garbage collection before starting
        gc.collect()
        
        # Start monitoring resources
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=0.1)  # Reset CPU measurement
        
        # Time model loading
        model_load_start = time.time()
        
        # Load the model - handle both model names and paths
        try:
            if os.path.exists(model_name):
                # It's a path
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
            else:
                # It's a model name
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Error loading Faster-Whisper model: {e}")
            return None
            
        #model_load_time = time.time() - model_load_start
        
        # Time first inference (latency)
        transcribe_start = time.time()
        try:
            segments_iterator, info = model.transcribe(audio_file, beam_size=5)
            
            # Get the first segment to measure latency
            first_segment_start = time.time()
            first_segment = next(segments_iterator, None)
            first_segment_time = time.time() - first_segment_start
            latency = first_segment_time if first_segment else None
            
            # Process remaining segments
            segments = [first_segment] if first_segment else []
            remaining_segments = list(segments_iterator)
            segments.extend(remaining_segments)
        except Exception as e:
            print(f"Error during transcription: {e}")
            segments = []
            info = None
            latency = None
        
        # Calculate total processing time
        total_time = time.time() - transcribe_start
        
        # Get CPU usage
        cpu_percent_end = psutil.cpu_percent(interval=None)
        
        # Write output
        if segments:
            with open(output_file, "w") as f:
                for segment in segments:
                    if segment:
                        f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
                    
        # Get memory usage after processing
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get audio duration
        audio_duration = self._get_audio_duration(audio_file)
        rtf = total_time / audio_duration if audio_duration else None
        
        # Extract transcription from segments for WER calculation
        transcription = " ".join(segment.text for segment in segments if segment)
        
        # Calculate WER if reference text is available
        error_rate = None
        audio_base = os.path.splitext(audio_name)[0]
        if HAS_JIWER and audio_base in self.reference_texts:
            error_rate = wer(self.reference_texts[audio_base], transcription)
        
        # Store results
        language = info.language if info else None
        language_prob = info.language_probability if info else None
        
        result_key = f"faster_whisper_{model_name}_{audio_name}"
        self.results[result_key] = {
            "model": f"Faster-Whisper {model_name}",
            "audio": audio_name,
            "total_time": total_time,
            "rtf": rtf,
            "wer": error_rate,
            #"memory_usage_mb": end_memory - start_memory,
            #"cpu_usage": cpu_percent_end,
            #"model_load_time": model_load_time,
            "latency": latency,
            "transcription": transcription,
            "language": language,
            "language_probability": language_prob
        }
        
        # Clean up
        del model
        gc.collect()
        
        return self.results[result_key]
        
        
    def measure_vosk(self, model_path, audio_file):
        """Benchmark Vosk"""
        model_name = os.path.basename(model_path) if model_path else "default"
        audio_name = os.path.basename(audio_file)
        print(f"Benchmarking Vosk with {model_name} on {audio_name}")
        
        # Prepare output file
        output_file = os.path.join(self.output_dir, f"vosk_{model_name}_{audio_name}.txt")
        
        # Start monitoring resources
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=0.1)  # Reset CPU measurement
        
        # Time model loading and processing
        start_time = time.time()
        
        # Build command based on whether a model path is provided
        if model_path and model_path.endswith('.zip'):
            # Extract the model if it's a zip file
            model_dir = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).replace('.zip', ''))
            
            # Check if the model is already extracted
            if not os.path.exists(model_dir):
                print(f"Extracting Vosk model to {model_dir}")
                import zipfile
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    os.makedirs(model_dir, exist_ok=True)
                    zip_ref.extractall(model_dir)
            
            # Use the extracted model
            cmd = [
                "vosk-transcriber", 
                "-m", model_dir,
                "-i", audio_file, 
                "-o", output_file
            ]
        else:
            # Use default model
            cmd = [
                "vosk-transcriber", 
                "-i", audio_file, 
                "-o", output_file
            ]
        
        # Determine if we need to activate virtual env
        vosk_env = os.getenv("VOSK_ENV_PATH", "./test_audio_py")
        
        if os.path.exists(vosk_env):
            # Run with virtual environment
            activate_script = os.path.join(vosk_env, "bin", "activate")
            if os.path.exists(activate_script):
                full_cmd = f"source {activate_script} && {' '.join(cmd)}"
                print(f"Cmd: {full_cmd}")
                result = subprocess.run(full_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)
            else:
                print(f"Cmd: {cmd}")
                result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            # Run without virtual environment
            print(f"Cmd: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        cpu_percent_end = psutil.cpu_percent(interval=None)
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        
        # Get audio duration
        audio_duration = self._get_audio_duration(audio_file)
        rtf = total_time / audio_duration if audio_duration else None
        
        # Extract transcription from output file
        transcription = ""
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                transcription = f.read()
        
        # Calculate WER if reference text is available
        error_rate = None
        audio_base = os.path.splitext(audio_name)[0]
        if HAS_JIWER and audio_base in self.reference_texts:
            error_rate = wer(self.reference_texts[audio_base], transcription)
        
        # Store results
        result_key = f"vosk_{model_name}_{audio_name}"
        self.results[result_key] = {
            "model": f"Vosk {model_name}",
            "audio": audio_name,
            "total_time": total_time,
            "rtf": rtf,
            "wer": error_rate,
            #"memory_usage_mb": end_memory - start_memory,
            #"cpu_usage": cpu_percent_end,
            #"model_load_time": None,  # Not easily extracted from Vosk
            "transcription": transcription
        }
        
        # Log any errors or output
        if result.returncode != 0:
            print(f"Vosk error: {result.stderr}")
        
        return self.results[result_key]
    
    
    
    
    def _get_audio_duration(self, audio_file):
        """Get audio duration using ffprobe"""
        try:
            # Check if ffprobe is available
            try:
                subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("ffprobe not found. Audio duration calculation will be disabled.")
                return None
                
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return None
    
    def export_results(self):
        """Export results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Copy to standard name for ease of access
        standard_json = os.path.join(self.output_dir, "benchmark_results.json")
        shutil.copy(json_path, standard_json)
        
        # Convert to DataFrame and save as CSV
        rows = []
        for key, result in self.results.items():
            # Skip the transcription text for CSV (it can be long)
            result_for_csv = {k: v for k, v in result.items() if k != 'transcription'}
            rows.append(result_for_csv)
        
        df = pd.DataFrame(rows)
        
        # Save timestamped CSV
        csv_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Copy to standard name for ease of access
        standard_csv = os.path.join(self.output_dir, "benchmark_results.csv")
        shutil.copy(csv_path, standard_csv)
        
        print(f"Results saved to {json_path} and {csv_path}")
        
        return df
    
    def generate_charts(self, df=None):
        """Generate comparison charts"""
        if df is None:
            # Convert results to DataFrame
            rows = []
            for key, result in self.results.items():
                # Skip the transcription text for charts
                result_for_chart = {k: v for k, v in result.items() if k != 'transcription'}
                rows.append(result_for_chart)
            df = pd.DataFrame(rows)
        
        # Create a directory for charts
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Group by model and audio file
        models = df['model'].unique()
        audio_files = df['audio'].unique()
        
        # Plot RTF comparison
        self._create_bar_chart(
            df, 'rtf', 'Real-Time Factor (RTF) Comparison', 
            'RTF (lower is better)', charts_dir, 'rtf_comparison.png'
        )
        
        # Plot WER comparison if available
        if 'wer' in df.columns and not df['wer'].isna().all():
            self._create_bar_chart(
                df, 'wer', 'Word Error Rate (WER) Comparison', 
                'WER (lower is better)', charts_dir, 'wer_comparison.png'
            )
        
        
        
        
        # Plot latency if available
        if 'latency' in df.columns and not df['latency'].isna().all():
            latency_df = df[~df['latency'].isna()]
            if not latency_df.empty:
                self._create_bar_chart(
                    latency_df, 'latency', 'Latency Comparison (Time to First Word)', 
                    'Latency (seconds)', charts_dir, 'latency_comparison.png'
                )
        
        # Plot model loading time if available
        if 'model_load_time' in df.columns and not df['model_load_time'].isna().all():
            load_df = df[~df['model_load_time'].isna()]
            if not load_df.empty:
                # Group by model and take the first occurrence for each model
                load_df = load_df.groupby('model', as_index=False).first()
                
                plt.figure(figsize=(12, 8))
                plt.bar(load_df['model'], load_df['model_load_time'])
                plt.title('Model Loading Time Comparison')
                plt.xlabel('Model')
                plt.ylabel('Loading Time (seconds)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, 'loading_time_comparison.png'))
                plt.close()
        
        # Create summary chart with multiple metrics
        self._create_summary_chart(df, charts_dir)
        
        print(f"Charts saved to {charts_dir}")
    
    def _create_bar_chart(self, df, metric, title, ylabel, output_dir, filename):
        """Create a bar chart for a specific metric"""
        plt.figure(figsize=(12, 8))
        
        # Group by audio file and model
        for audio in df['audio'].unique():
            subset = df[df['audio'] == audio]
            if not subset[metric].isna().all():
                plt.bar(subset['model'], subset[metric], label=audio)
        
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def _create_summary_chart(self, df, output_dir):
        """Create a summary chart with multiple metrics"""
        # Select metrics to visualize
        metrics = ['rtf']
        if 'wer' in df.columns and not df['wer'].isna().all():
            metrics.append('wer')
        
        
        # For each audio file, create a chart
        for audio in df['audio'].unique():
            audio_df = df[df['audio'] == audio]
            
            # Skip if empty
            if audio_df.empty:
                continue
                
            plt.figure(figsize=(14, 10))
            
            # Create subplots for each metric
            n_metrics = len(metrics)
            for i, metric in enumerate(metrics):
                plt.subplot(n_metrics, 1, i+1)
                
                valid_data = audio_df[~audio_df[metric].isna()]
                if not valid_data.empty:
                    plt.bar(valid_data['model'], valid_data[metric])
                    
                    # Set title and labels
                    if metric == 'rtf':
                        plt.title(f'Real-Time Factor (lower is better) - {audio}')
                        plt.ylabel('RTF')
                    elif metric == 'wer':
                        plt.title(f'Word Error Rate (lower is better) - {audio}')
                        plt.ylabel('WER')
                    
                    
                    plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'summary_{os.path.splitext(audio)[0]}.png'))
            plt.close()

    def create_report(self):
        """Create a markdown report with results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(self.output_dir, "benchmark_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Speech Recognition Model Benchmark Report\n\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            # Summary of models benchmarked
            f.write("## Models Benchmarked\n\n")
            
            models_by_type = {}
            for key, result in self.results.items():
                model_type = result['model'].split()[0]  # WhisperCPP, Vosk, Faster-Whisper
                model_name = result['model']
                
                if model_type not in models_by_type:
                    models_by_type[model_type] = set()
                    
                models_by_type[model_type].add(model_name)
            
            for model_type, models in models_by_type.items():
                f.write(f"- {model_type}: {len(models)} models\n")
            
            # Audio files
            audio_files = set(result['audio'] for result in self.results.values())
            f.write(f"\n## Audio Files\n\n")
            for audio in audio_files:
                duration = self._get_audio_duration(os.path.join("sample_audio", audio))
                duration_str = f"{duration:.2f} seconds" if duration else "Unknown duration"
                f.write(f"- {audio} ({duration_str})\n")
            
            # Best performers
            f.write("\n## Performance Summary\n\n")
            
            # Create DataFrame for analysis
            rows = []
            for key, result in self.results.items():
                # Extract relevant metrics
                row = {
                    'model': result['model'],
                    'audio': result['audio'],
                    'rtf': result.get('rtf'),
                    'wer': result.get('wer'),
                    #'memory_usage_mb': result.get('memory_usage_mb'),
                    'total_time': result.get('total_time')
                }
                rows.append(row)
            
            analysis_df = pd.DataFrame(rows)
            
            # Find best model overall for RTF
            if not analysis_df['rtf'].isna().all():
                best_rtf = analysis_df.loc[analysis_df['rtf'].idxmin()]
                f.write(f"- **Fastest Model Overall**: {best_rtf['model']} (RTF: {best_rtf['rtf']:.4f})\n")
            
            # Find best model for WER if available
            if 'wer' in analysis_df.columns and not analysis_df['wer'].isna().all():
                wer_df = analysis_df[~analysis_df['wer'].isna()]
                if not wer_df.empty:
                    best_wer = wer_df.loc[wer_df['wer'].idxmin()]
                    f.write(f"- **Most Accurate Model**: {best_wer['model']} (WER: {best_wer['wer']:.4f})\n")
            
            
            
            # Results tables
            f.write("\n## Detailed Results\n\n")
            
            # For each audio file, create a table
            for audio in audio_files:
                f.write(f"### Results for {audio}\n\n")
                
                audio_results = analysis_df[analysis_df['audio'] == audio]
                
                # Create markdown table
                f.write("| Model | RTF | Processing Time | ")
                if 'wer' in audio_results.columns and not audio_results['wer'].isna().all():
                    f.write("WER | ")
                
                
                f.write("| --- | --- | --- | ")
                if 'wer' in audio_results.columns and not audio_results['wer'].isna().all():
                    f.write("--- | ")
                
                
                # Sort by RTF
                audio_results = audio_results.sort_values('rtf')
                
                for _, row in audio_results.iterrows():
                    rtf = f"{row['rtf']:.4f}" if not pd.isna(row['rtf']) else "N/A"
                    time = f"{row['total_time']:.2f}s" if not pd.isna(row['total_time']) else "N/A"
                    #mem = f"{row['memory_usage_mb']:.2f}" if not pd.isna(row['memory_usage_mb']) else "N/A"
                    
                    f.write(f"| {row['model']} | {rtf} | {time} | ")
                    
                    if 'wer' in row and not pd.isna(row['wer']):
                        wer_val = f"{row['wer']:.4f}"
                        f.write(f"{wer_val} | ")
                        
                    #f.write(f"{mem} |\n")
                
                f.write("\n")
            
            # Visualizations
            charts_dir = os.path.join(self.output_dir, "charts")
            if os.path.exists(charts_dir) and os.listdir(charts_dir):
                f.write("\n## Visualizations\n\n")
                f.write("Charts are available in the `benchmark_results/charts/` directory:\n\n")
                
                for chart_file in os.listdir(charts_dir):
                    f.write(f"- [{chart_file}](charts/{chart_file})\n")
        
        print(f"Report saved to {report_path}")
        return report_path

def create_reference_files(audio_dir, reference_dir="reference_transcriptions"):
    """Create template reference files for the user to fill in"""
    os.makedirs(reference_dir, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            audio_files.append(filename)
    
    # Create template files
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        ref_file = os.path.join(reference_dir, f"{base_name}_reference.txt")
        
        if not os.path.exists(ref_file):
            with open(ref_file, 'w') as f:
                f.write(f"# Reference transcription for {audio_file}\n")
                f.write("# Replace this text with the accurate transcription of the audio file\n")
                f.write("# This will be used to calculate Word Error Rate (WER)\n")
                f.write("# Lines starting with # are comments and will be ignored\n\n")

def find_models(base_dir="."):
    """Find available models in the directory structure"""
    models = {
        "whispercpp": [],
        "faster_whisper": [],
        "vosk": []
    }
    
    # Find WhisperCPP models
    whispercpp_dir = os.path.join(base_dir, "models", "whispercpp_models")
    if os.path.exists(whispercpp_dir):
        for file in os.listdir(whispercpp_dir):
            if file.endswith('.bin'):
                models["whispercpp"].append(os.path.join(whispercpp_dir, file))
    
    # Find Faster Whisper models
    # For faster-whisper, we'll list the model names rather than paths
    faster_whisper_dir = os.path.join(base_dir, "models", "faster_whisper_models")
    if os.path.exists(faster_whisper_dir):
        for dirname in os.listdir(faster_whisper_dir):
            if dirname.startswith("models--Systran--faster-whisper-"):
                # Extract model name from directory name
                model_name = dirname.replace("models--Systran--faster-whisper-", "")
                models["faster_whisper"].append(model_name)
    
    # Find Vosk models
    vosk_dir = os.path.join(base_dir, "models", "vosk_models")
    if os.path.exists(vosk_dir):
        for file in os.listdir(vosk_dir):
            if file.endswith('.zip'):
                models["vosk"].append(os.path.join(vosk_dir, file))
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Benchmark speech recognition models')
    parser.add_argument('--audio_dir', default='./sample_audio', help='Directory containing audio files')
    parser.add_argument('--ref_dir', default='./reference_transcriptions', help='Directory containing reference transcriptions')
    parser.add_argument('--output_dir', default='./benchmark_results', help='Directory to store results')
    parser.add_argument('--create_references', action='store_true', help='Create template reference transcription files')
    parser.add_argument('--whispercpp', action='store_true', help='Run WhisperCPP benchmarks')
    parser.add_argument('--faster_whisper', action='store_true', help='Run Faster-Whisper benchmarks')
    parser.add_argument('--vosk', action='store_true', help='Run Vosk benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all available models')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    parser.add_argument('--specific_models', nargs='+', help='Run specific models (provide full paths)')
    
    args = parser.parse_args()
    
    # Find available models
    available_models = find_models()
    
    # List models if requested
    if args.list_models:
        print("\nAvailable models:")
        print("\nWhisperCPP models:")
        for model in available_models["whispercpp"]:
            print(f"  - {model}")
        
        print("\nFaster-Whisper models:")
        for model in available_models["faster_whisper"]:
            print(f"  - {model}")
        
        print("\nVosk models:")
        for model in available_models["vosk"]:
            print(f"  - {model}")
        
        return
    
    # Create reference files if requested
    if args.create_references:
        create_reference_files(args.audio_dir, args.ref_dir)
        print(f"Created template reference files in {args.ref_dir}")
        return
    
    # Create benchmark instance
    benchmark = ModelBenchmark(args.output_dir)
    
    # Load any reference transcriptions
    benchmark.load_references(args.ref_dir)
    
    # Get list of audio files
    audio_files = []
    for filename in os.listdir(args.audio_dir):
        if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            audio_files.append(os.path.join(args.audio_dir, filename))
    
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return
    
    models_to_run = {
        "whispercpp": [],
        "faster_whisper": [],
        "vosk": []
    }
    
    # Determine which models to run
    if args.all:
        # Run all available models
        models_to_run = available_models
    else:
        # Run specific types of models
        if args.whispercpp:
            models_to_run["whispercpp"] = available_models["whispercpp"]
        
        if args.faster_whisper:
            models_to_run["faster_whisper"] = available_models["faster_whisper"]
        
        if args.vosk:
            models_to_run["vosk"] = available_models["vosk"]
        
        # Add specific models if provided
        if args.specific_models:
            for model_path in args.specific_models:
                if model_path.endswith('.bin') and os.path.exists(model_path):
                    models_to_run["whispercpp"].append(model_path)
                elif model_path.endswith('.zip') and os.path.exists(model_path):
                    models_to_run["vosk"].append(model_path)
                elif model_path in ["tiny.en", "base.en", "small.en", "medium.en"]:
                    models_to_run["faster_whisper"].append(model_path)
    
    # Run benchmarks
    benchmarks_run = 0
    
    # WhisperCPP benchmarks
    for model_path in models_to_run["whispercpp"]:
        for audio_file in audio_files:
            try:
                print(f"\nRunning WhisperCPP: {os.path.basename(model_path)} on {os.path.basename(audio_file)}")
                print(f"\nModel Path:{model_path}")
                benchmark.measure_whispercpp(model_path, audio_file)
                benchmarks_run += 1
            except Exception as e:
                print(f"Error running WhisperCPP {model_path} on {audio_file}: {e}")
    
    # Faster-Whisper benchmarks
    for model_name in models_to_run["faster_whisper"]:
        for audio_file in audio_files:
            try:
                print(f"\nRunning Faster-Whisper: {model_name} on {os.path.basename(audio_file)}")
                benchmark.measure_faster_whisper(model_name, audio_file)
                benchmarks_run += 1
            except Exception as e:
                print(f"Error running Faster-Whisper {model_name} on {audio_file}: {e}")
    
    # Vosk benchmarks
    for model_path in models_to_run["vosk"]:
        for audio_file in audio_files:
            try:
                print(f"\nRunning Vosk: {os.path.basename(model_path)} on {os.path.basename(audio_file)}")
                benchmark.measure_vosk(model_path, audio_file)
                benchmarks_run += 1
            except Exception as e:
                print(f"Error running Vosk {model_path} on {audio_file}: {e}")
    
    # Check if any benchmarks were run
    if benchmarks_run == 0:
        print("\nNo benchmarks were run. Use --all to run all available models, or specify models with --whispercpp, --faster_whisper, --vosk, or --specific_models.")
        return
    
    # Export results
    benchmark.export_results()
    
    # Generate charts
    benchmark.generate_charts()
    
    # Create report
    report_path = benchmark.create_report()
    
    print(f"\nBenchmark completed. {benchmarks_run} benchmarks run.")
    print(f"Results saved to {args.output_dir}")
    print(f"Report: {report_path}")
    print(f"Charts: {os.path.join(args.output_dir, 'charts')}")

if __name__ == "__main__":
    main()
