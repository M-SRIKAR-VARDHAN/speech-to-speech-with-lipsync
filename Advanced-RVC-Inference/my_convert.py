import os, sys, argparse, torch, librosa, soundfile as sf
import numpy as np
import logging
import warnings
from contextlib import contextmanager
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from config import Config
import fairseq.data
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
def main():
    # --- ARGS ---
    parser = argparse.ArgumentParser(description="RVC Voice Conversion")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth voice model file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output audio file.")
    parser.add_argument("--index", type=str, default="", help="Path to the .index feature file. (Optional)")
    parser.add_argument("--pitch", type=int, default=0, help="Transpose pitch in semitones.")
    parser.add_argument("--f0_method", type=str, default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"], help="Pitch extraction method.")
    parser.add_argument("--index_rate", type=float, default=0.7, help="Ratio of feature retrieval.")
    parser.add_argument("--filter_radius", type=int, default=3, help="Median filtering radius for pitch.")
    parser.add_argument("--resample_sr", type=int, default=0, help="Output sample rate (0 = keep target sr).")
    parser.add_argument("--rms_mix_rate", type=float, default=1.0, help="Volume envelope mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protects voiceless consonants.")
    args = parser.parse_args()

    print("=" * 40)
    print("🚀 Starting RVC Voice Conversion...")
    print("=" * 40)
    
    try:
        # --- CONFIG ---
        with suppress_stdout_stderr():
            config = Config()
            vc_pipeline = VC(tgt_sr=16000, config=config)

        # --- LOAD HUBERT ---
        print("-> [1/5] Loading Hubert model...")
        with suppress_stdout_stderr():
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"])
            hubert_model = models[0].to(config.device)
            hubert_model.eval()
        print("   - Hubert model loaded.")

        # --- LOAD VOICE MODEL ---
        print(f"-> [2/5] Loading voice model: {os.path.basename(args.model)}")
        cpt = torch.load(args.model, map_location="cpu")
        
        # --- DETERMINE SAMPLE RATE ---
        model_sr = cpt.get("sr")
        if model_sr:
            model_sr = int(str(model_sr).replace("k", "")) * 1000 if "k" in str(model_sr) else int(model_sr)
        else: # Fallback if sr is not in the checkpoint
            model_sr = 48000
        
        version = cpt.get("version", "v1")
        print(f"   - Voice model loaded (Version: {version}, SR: {model_sr}Hz).")
        
        with suppress_stdout_stderr():
            net_g = vc_pipeline.get_vc(cpt, version)

        # --- LOAD FAISS INDEX (if provided) ---
        if args.index and os.path.exists(args.index):
            print(f"-> [3/5] Loading FAISS index...")
            from faiss import read_index
            index = read_index(args.index)
            print(f"   - FAISS index loaded (Entries: {index.ntotal}).")
        else:
            print("-> [3/5] No FAISS index provided, skipping.")


        # --- LOAD INPUT AUDIO ---
        print(f"-> [4/5] Loading input audio: {os.path.basename(args.input)}")
        audio_hubert, _ = librosa.load(args.input, sr=16000, mono=True)
        print("   - Audio loaded and pre-processed.")

        # --- RUN INFERENCE PIPELINE ---
        print("-> [5/5] Performing voice conversion...")
        with suppress_stdout_stderr():
            out_audio = vc_pipeline.pipeline(
                model=hubert_model,
                net_g=net_g,
                sid=0,
                audio=audio_hubert,
                input_audio_path=args.input,
                times=[0,0,0],
                f0_up_key=args.pitch,
                f0_method=args.f0_method,
                file_index=args.index,
                index_rate=args.index_rate,
                if_f0=1,
                filter_radius=args.filter_radius,
                tgt_sr=model_sr,
                resample_sr=args.resample_sr if args.resample_sr > 0 else model_sr,
                rms_mix_rate=args.rms_mix_rate,
                version=version,
                protect=args.protect,
            )
        print("   - Conversion complete.")

        # --- SAVE OUTPUT ---
        output_sr = args.resample_sr if args.resample_sr > 0 else model_sr
        sf.write(args.output, out_audio, output_sr)
        
        print("\n" + "=" * 40)
        print("🎉 Conversion Finished Successfully!")
        print(f"   -> Output saved to: {os.path.abspath(args.output)}")
        print("=" * 40)

    except Exception as e:
        print("\n" + "=" * 40)
        print(f"❌ ERROR: A fatal error occurred during the conversion process.")
        # To see the actual error for debugging, you can uncomment the next line
        # import traceback; traceback.print_exc()
        print(f"   - Details: {e}")
        print("=" * 40)
        sys.exit(1)

if __name__ == "__main__":
    main()


