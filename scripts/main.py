import os
import subprocess
from pathlib import Path
FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"

def convert_wav_to_mp3(source_dir: str, output_dir_name: str = "test 2") -> None:
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Folder '{source_dir}' not found.")
        return

    output_path = source_path / output_dir_name
    output_path.mkdir(exist_ok=True)

    wav_files = list(source_path.glob("*.wav"))
    if not wav_files:
        print(f"ℹ️ No .wav files in folder '{source_dir}'.")
        return

    print(f"Found {len(wav_files)} WAV file(s). Starting conversion...\n")

    for wav_file in wav_files:
        mp3_file = output_path / (wav_file.stem + ".mp3")
        cmd = [
            FFMPEG_EXE,
            "-i", str(wav_file),
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            str(mp3_file)
        ]
        try:
            print(f"🔄 Converting: {wav_file.name} → {mp3_file.name}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Done\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {wav_file.name}:")
            print(f"  Exit code: {e.returncode}")
            print("  FFmpeg message:")
            print(e.stderr if e.stderr else e.stdout)
        except FileNotFoundError:
            print(f"❌ FFmpeg not found. Make sure '{FFMPEG_EXE}' is accessible.")
            print("  Install FFmpeg or set the correct path in FFMPEG_EXE.")
            return

    print(f"✔️ Processing complete. Files saved to: {output_path}")

if __name__ == "__main__":
    source = os.path.expanduser(r"~\f5-tts-project\tests")
    convert_wav_to_mp3(source)
