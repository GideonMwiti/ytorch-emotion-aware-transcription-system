"""
Interactive Emotion-Aware Speech Transcription CLI Tool
======================================================
Production command-line interface for transcribing audio recordings
with speech emotion tags and valence-arousal affect scores.

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import soundfile as sf
import torch
import torchaudio.transforms as T

from main import (
    Wav2Vec2ASREngine,
    EmotionCNN_GRU,
    SileroVADSegmenter,
    EmotionTranscriptionPipeline,
    SAMPLE_RATE,
    EMOTIONS,
    NUM_CLASSES,
    VAD_MODEL_PATH,
    EMOTION_MODEL_PATH,
    OUTPUTS_DIR
)

def render_ascii_meter(prob: float, width: int = 16) -> str:
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}|"

def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def format_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds % 60)
    m = int((seconds // 60) % 60)
    h = int(seconds // 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    parser = argparse.ArgumentParser(description="Emotion-Aware Speech Transcription CLI Tool")
    parser.add_argument("--audio", type=str, help="Path to input audio file")
    parser.add_argument("--threshold", type=float, default=0.50, help="VAD trigger threshold (default 0.50)")
    parser.add_argument("--export-txt", type=str, help="Export plaintext transcript")
    parser.add_argument("--export-json", type=str, help="Export JSON transcript")
    parser.add_argument("--export-srt", type=str, help="Export SubRip (SRT) subtitles")

    args = parser.parse_args()

    audio_path = args.audio
    if not audio_path:
        default_path = os.path.join(OUTPUTS_DIR, "emotional_meeting.wav")
        if os.path.exists(default_path):
            audio_path = default_path
        else:
            parser.print_help()
            sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("      EMOTION-AWARE TRANSCRIPTION SYSTEM (ASR + SPEECH EMOTION)")
    print("=" * 78)
    print(f"Input Audio File : {audio_path}")
    print(f"Target Emotions  : {', '.join(EMOTIONS)}")

    # Load audio
    wav, sr = sf.read(audio_path)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)
    if wav_tensor.ndim > 1:
        wav_tensor = wav_tensor.mean(dim=-1)
    if sr != SAMPLE_RATE:
        wav_tensor = T.Resample(sr, SAMPLE_RATE)(wav_tensor.unsqueeze(0)).squeeze(0)

    duration_s = len(wav_tensor) / SAMPLE_RATE
    print(f"Audio Duration   : {duration_s:.2f} seconds ({len(wav_tensor):,} samples)")
    print("-" * 78)

    # Load models
    print("Loading models (Silero VAD + Wav2Vec2 CTC + EmotionCNN_GRU)...")
    vad_segmenter = SileroVADSegmenter(VAD_MODEL_PATH)
    asr_engine = Wav2Vec2ASREngine()

    emotion_model = EmotionCNN_GRU(num_classes=NUM_CLASSES)
    if os.path.exists(EMOTION_MODEL_PATH):
        emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location="cpu"))
    emotion_model.eval()

    pipeline = EmotionTranscriptionPipeline(asr_engine, emotion_model, vad_segmenter)

    # Process
    t_start = time.perf_counter()
    aligned_turns, timing = pipeline.process(wav_tensor)
    total_time = time.perf_counter() - t_start
    rtf = total_time / duration_s if duration_s > 0 else 0.0

    print("\n" + "=" * 78)
    print("       EMOTION-ATTRIBUTED TRANSCRIPT ('WHO SAID WHAT WITH WHAT EMOTION')")
    print("=" * 78)

    transcript_lines = []
    for turn in aligned_turns:
        st_str = format_timestamp(turn["start"])
        et_str = format_timestamp(turn["end"])
        meter = render_ascii_meter(turn["confidence"], width=14)
        line_display = f"[{st_str} - {et_str}] [{turn['emotion']:<9} {meter} {turn['confidence']*100:4.1f}%] {turn['text']}"
        transcript_lines.append(f"[{st_str} - {et_str}] [{turn['emotion']} ({turn['confidence']*100:.1f}%)] {turn['text']}")
        print(line_display)

    print("-" * 78)
    print("PIPELINE SUMMARY & BENCHMARK:")
    print(f"  • Dialogue Turns Transcribed: {len(aligned_turns)}")
    print(f"  • Dominant Emotion Found    : {max(set(t['emotion'] for t in aligned_turns), key=[t['emotion'] for t in aligned_turns].count) if aligned_turns else 'N/A'}")
    print(f"  • Total Audio Duration      : {duration_s:.2f} s")
    print(f"  • Total Processing Time     : {total_time:.2f} s (RTF: {rtf:.4f}x, {1.0/rtf:.1f}x speedup)")
    print("=" * 78)

    # Export
    if args.export_txt:
        with open(args.export_txt, "w") as f:
            f.write("\n".join(transcript_lines) + "\n")
        print(f"Plaintext transcript exported to: {args.export_txt}")

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(aligned_turns, f, indent=2)
        print(f"JSON transcript exported to: {args.export_json}")

    if args.export_srt:
        with open(args.export_srt, "w") as f:
            for idx, turn in enumerate(aligned_turns):
                f.write(f"{idx + 1}\n")
                f.write(f"{format_srt_time(turn['start'])} --> {format_srt_time(turn['end'])}\n")
                f.write(f"[{turn['emotion']}] {turn['text']}\n\n")
        print(f"SRT subtitles exported to: {args.export_srt}")

if __name__ == "__main__":
    main()

