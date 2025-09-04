import argparse
import os
import time
from pathlib import Path
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel
import scipy.io.wavfile
from moviepy.editor import VideoFileClip
import traceback  # <-- ADDED: Import the traceback module

class Config:
    WHISPER_MODEL_PATH = Path("./models/whisper")
    TRANSLATION_MODEL_PATH = Path("./models/translation/nllb-en-te")
    TTS_MODEL_PATH = Path("./models/tts/mms-tel")
    WHISPER_MODEL_NAME = "base.en"
    SRC_LANG = "eng_Latn"
    TGT_LANG = "tel_Telu"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEMP_AUDIO_FILENAME = "temp_extracted_audio_for_pipeline.wav"

def extract_audio_from_video(video_path: str, audio_output_path: str):
    print(f"\n[STEP 1/4] 🎬 Extracting audio from '{os.path.basename(video_path)}'...")
    start_time = time.time()
    try:
        video_clip = VideoFileClip(video_path)
        # Suppress moviepy's console output
        video_clip.audio.write_audiofile(audio_output_path, codec='pcm_s16le', logger=None)
        video_clip.close()
        duration = time.time() - start_time
        print(f"Audio extracted successfully to '{audio_output_path}' in {duration:.2f}s.")
        return True
    except Exception as e:
        print(f"❌ ERROR during audio extraction: {e}")
        # --- ADDED: Print full traceback for detailed debugging ---
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------\n")
        # --- END ADDITION ---
        return False

def transcribe_audio(audio_path: str, cfg: Config) -> str:
    print(f"\n[STEP 2/4] 🎤 Transcribing audio with Whisper ({cfg.WHISPER_MODEL_NAME})...")
    start_time = time.time()
    try:
        model = whisper.load_model(cfg.WHISPER_MODEL_NAME, device=cfg.DEVICE, download_root=str(cfg.WHISPER_MODEL_PATH))
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        transcribed_text = result["text"].strip()
        duration = time.time() - start_time
        print(f"Transcription complete in {duration:.2f}s.")
        print(f'   -> English Text: "{transcribed_text}"')
        return transcribed_text
    except Exception as e:
        print(f"❌ ERROR during transcription: {e}")
        # --- ADDED: Print full traceback for detailed debugging ---
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------\n")
        # --- END ADDITION ---
        return None

def translate_text(text_to_translate: str, cfg: Config) -> str:
    print(f"\n[STEP 3/4] 🌐 Translating text from '{cfg.SRC_LANG}' to '{cfg.TGT_LANG}'...")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TRANSLATION_MODEL_PATH, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.TRANSLATION_MODEL_PATH).to(cfg.DEVICE)

        tokenizer.src_lang = cfg.SRC_LANG
        inputs = tokenizer(text_to_translate, return_tensors="pt").to(cfg.DEVICE)

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(cfg.TGT_LANG),
            max_length=512  # Increased max_length for longer text
        )

        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        duration = time.time() - start_time
        print(f"Translation complete in {duration:.2f}s.")
        print(f'   -> Telugu Text: "{translated_text}"')
        return translated_text
    except Exception as e:
        print(f"❌ ERROR during translation: {e}")
        # --- ADDED: Print full traceback for detailed debugging ---
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------\n")
        # --- END ADDITION ---
        return None

def synthesize_speech(text_to_speak: str, output_path: str, cfg: Config):
    print(f"\n[STEP 4/4] 🎵 Synthesizing speech to '{output_path}'...")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TTS_MODEL_PATH)
        model = VitsModel.from_pretrained(cfg.TTS_MODEL_PATH).to(cfg.DEVICE)

        inputs = tokenizer(text_to_speak, return_tensors="pt").to(cfg.DEVICE)

        with torch.no_grad():
            output = model(**inputs).waveform

        speech_array = output.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate

        scipy.io.wavfile.write(output_path, rate=sample_rate, data=speech_array)
        duration = time.time() - start_time
        print(f"✅ Speech synthesized successfully in {duration:.2f}s.")
        return True
    except Exception as e:
        print(f"❌ ERROR during speech synthesis: {e}")
        # --- ADDED: Print full traceback for detailed debugging ---
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------\n")
        # --- END ADDITION ---
        return False


def main():
    parser = argparse.ArgumentParser(description="A complete pipeline to convert video to translated audio.")
    parser.add_argument("--video_file", required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--output_file", default="translated_output.wav", help="Path for the final translated WAV audio file.")
    args = parser.parse_args()

    if not os.path.exists(args.video_file):
        print(f"❌ FATAL: Input video file not found at '{args.video_file}'")
        return

    cfg = Config()

    print("=" * 70)
    print("🚀 Starting Video-to-Translated-Audio Pipeline")
    print(f"   - Device: {cfg.DEVICE}")
    print(f"   - Input Video: {args.video_file}")
    print(f"   - Final Output: {args.output_file}")
    print("=" * 70)

    overall_start_time = time.time()
    try:

        if not extract_audio_from_video(args.video_file, cfg.TEMP_AUDIO_FILENAME):
            return

        english_text = transcribe_audio(cfg.TEMP_AUDIO_FILENAME, cfg)
        if not english_text:
            return

        telugu_text = translate_text(english_text, cfg)
        if not telugu_text:
            return

        if not synthesize_speech(telugu_text, args.output_file, cfg):
            return

    finally:
        if os.path.exists(cfg.TEMP_AUDIO_FILENAME):
            os.remove(cfg.TEMP_AUDIO_FILENAME)
            print(f"\n🧹 Cleaned up temporary file: {cfg.TEMP_AUDIO_FILENAME}")

    overall_duration = time.time() - overall_start_time
    print("\n" + "=" * 70)
    print(f"🎉🎉🎉 Pipeline Completed Successfully! 🎉🎉🎉")
    print(f"   -> Final audio file is ready at: {os.path.abspath(args.output_file)}")
    print(f"   -> Total execution time: {overall_duration:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
