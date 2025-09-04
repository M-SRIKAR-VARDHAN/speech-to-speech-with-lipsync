import os
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command, streaming its output"""
    process = subprocess.Popen(cmd, shell=True, cwd=cwd)
    process.communicate()
    if process.returncode != 0:
        print(f"❌ Command failed with return code {process.returncode}: {cmd}")
        raise RuntimeError(f"Command failed: {cmd}")

def main(video_file, final_output_filename):
    demo_dir = Path(__file__).parent.resolve()
    rvc_dir = demo_dir / "Advanced-RVC-Inference"

    # ============================ CHANGE 1 of 3 ============================
    # Create a full, absolute path for the final output file.
    # This prevents it from being saved in the wrong directory.
    final_output_path = demo_dir / final_output_filename

    base = Path(video_file).stem
    translated_audio = demo_dir / f"{base}_translated.wav"
    converted_audio = demo_dir / f"{base}_converted.wav"

    # --- Step 1: Run translation (1.py) ---
    print("🚀 Running translation pipeline...")
    run_command(
        f'python 1.py --video_file "{video_file}" --output_file "{translated_audio}"',
        cwd=demo_dir
    )

    # --- Step 2: Copy translated audio to RVC folder ---
    temp_input = rvc_dir / f"{base}_for_rvc.wav"
    shutil.copy(translated_audio, temp_input)

    # --- Step 3: Run RVC (my_convert.py) ---
    print("🎤 Running RVC voice conversion...")
    temp_output = rvc_dir / f"{base}_rvc.wav"
    run_command(
        f'python my_convert.py --model "weights/modi.pth" --index "weights/model.index" '
        f'--input "{temp_input.name}" --output "{temp_output.name}"',
        cwd=rvc_dir
    )

    # --- Step 4: Copy RVC result back to demo ---
    shutil.copy(temp_output, converted_audio)

    # --- Step 5: Replace video audio with RVC result ---
    print("🎬 Combining video and converted audio...")
    dubbed_video = demo_dir / f"dubbed_{base}.mp4"
    run_command(
        f'ffmpeg -y -i "{video_file}" -i "{converted_audio}" '
        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{dubbed_video}"',
        cwd=demo_dir
    )

    # --- Step 6: Run Lip Syncing (lip.py) ---
    print("👄 Running lip-syncing...")
    # ============================ CHANGE 2 of 3 ============================
    # We now pass the full, absolute path to the --output_video argument.
    run_command(
        f'python lip.py --input_video "{dubbed_video}" --output_video "{final_output_path}"',
    cwd=demo_dir
    )

    print("\n========================================")
    # ============================ CHANGE 3 of 3 ============================
    # Point to the absolute path in the final message.
    print(f"🎉 Final lip-synced dubbed video ready: {final_output_path}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", required=True, help="Input video file (e.g., test.mp4)")
    parser.add_argument("--output_file", required=True, help="Final output video file (e.g., final_video.mp4)")
    args = parser.parse_args()

    main(args.video_file, args.output_file)