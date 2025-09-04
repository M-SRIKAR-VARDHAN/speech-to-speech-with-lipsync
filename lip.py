import os
import subprocess
import argparse
from moviepy.editor import VideoFileClip

def preprocess_video(video_path, output_path, target_height=480):
    """
    Resizes the video to a standard resolution to prevent memory errors.
    The default height is now 720p.
    """
    print(f"\n--- Pre-processing Step: Resizing Video to {target_height}p ---")
    try:
        with VideoFileClip(video_path) as video_clip:
            # Resize the video clip while maintaining the aspect ratio
            resized_clip = video_clip.resize(height=target_height)
            resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=25, threads=4, preset="ultrafast", logger=None, verbose=False)

        print(f"✅ Video resized and saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error during video pre-processing: {e}")
        return False

def extract_audio(video_path, audio_output_path):
    """
    Extracts the audio from a video file and saves it as a WAV file.
    """
    print(f"--- Step 1: Extracting Audio from '{os.path.basename(video_path)}' ---")
    try:
        with VideoFileClip(video_path) as video_clip:
            video_clip.audio.write_audiofile(audio_output_path, codec="pcm_s16le", fps=16000)
        print(f"✅ Audio successfully extracted to: {audio_output_path}")
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False

def run_wav2lip(video_path, audio_path, output_path, checkpoint_path, box_coords):
    """
    Runs the Wav2Lip inference script to synchronize lip movements.
    """
    print("\n--- Step 2: Running Wav2Lip for Re-syncing ---")
    
    wav2lip_inference_script = os.path.join("wav2lip", "inference.py")

    if not os.path.exists(wav2lip_inference_script):
        print(f"❌ Error: Wav2Lip inference script not found at '{wav2lip_inference_script}'")
        return False

    command = [
        "python", wav2lip_inference_script,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--wav2lip_batch_size", "8",
        "--pads", "0", "10", "0", "0"
    ]

    if box_coords:
        command.extend(["--box", *[str(c) for c in box_coords]])

    print(f"▶️ Executing command: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True)
        print(f"✅ Lip-sync complete! Final video saved to: {output_path}")
        return True
    except FileNotFoundError:
        print("❌ Error: 'python' command not found. Please ensure Python is in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ An error occurred while running Wav2Lip.")
        print(f"   Return code: {e.returncode}")
        print(f"   Stderr: {e.stderr}")
        return False

def main():
    """
    Main function to orchestrate the video re-syncing pipeline.
    """
    parser = argparse.ArgumentParser(description="Re-sync a video's lip movements with its own audio using Wav2Lip.")
    parser.add_argument("--input_video", required=True, help="Path to the input video file.")
    parser.add_argument("--output_video", default="result_resynced.mp4", help="Path for the final re-synced video.")
    parser.add_argument("--wav2lip_checkpoint", default=os.path.join("wav2lip", "checkpoints", "wav2lip.pth"), help="Path to the Wav2Lip model checkpoint.")
    parser.add_argument("--box", type=int, nargs=4, help="Manually specify the face bounding box as X1 Y1 X2 Y2 to bypass the detector.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"❌ Error: Input video not found at '{args.input_video}'")
        return

    if not os.path.exists(args.wav2lip_checkpoint):
        print(f"❌ Error: Wav2Lip checkpoint not found at '{args.wav2lip_checkpoint}'")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define paths for our intermediate files
    resized_video_path = os.path.join(script_dir, "temp_resized_video.mp4")
    extracted_audio_path = os.path.join(script_dir, "extracted_audio.wav")

    # Pre-process Step: Resize video
    if not preprocess_video(args.input_video, resized_video_path):
        return

    # Step 1: Extract Audio from the ORIGINAL video to maintain quality
    if not extract_audio(args.input_video, extracted_audio_path):
        return

    # Step 2: Run Wav2Lip on the RESIZED video
    run_wav2lip(resized_video_path, extracted_audio_path, args.output_video, args.wav2lip_checkpoint, args.box)
    
    # Clean up all intermediate files
    for file_path in [resized_video_path, extracted_audio_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"\n🧹 Cleaned up intermediate file: {file_path}")


if __name__ == '__main__':
    main()