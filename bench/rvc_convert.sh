# Convert a vocal with a trained Applio voice, then lay it over a backing.
#   bash bench/rvc_convert.sh NAME vocal.wav backing.wav out.mp3 [pitch=0]
set -e
IN="$(cd "$(dirname "$2")" && pwd -W)/$(basename "$2")"; BK="$(cd "$(dirname "$3")" && pwd -W)/$(basename "$3")"; OUT="$(cd "$(dirname "$4")" && pwd -W)/$(basename "$4")"
R="$(cd "$(dirname "$0")/.." && pwd -W)"; cd "$R/models_ext/applio"
PY="$R/venv_applio/Scripts/python.exe"; N=$1; P=${5:-0}; export PYTHONUTF8=1
PTH=$(ls -t logs/$N/${N}_*e_*s.pth | head -1); IDX=$(ls logs/$N/*.index | grep -v trained | head -1)
echo "model $PTH index $IDX"
$PY core.py infer --input-path "$IN" --output-path "$R/outputs/rvc_${N}_vox.wav" \
  --pth-path "$(pwd -W)/$PTH" --index-path "$(pwd -W)/$IDX" --pitch $P --index-rate ${IR:-0.4} --protect ${PR:-0.5} --f0-method rmvpe --volume-envelope ${VE:-1} --export-format WAV
cd "$R"; "$R/venv/Scripts/python.exe" - "$R/outputs/rvc_${N}_vox.wav" "$BK" "$OUT" <<'PY'
import os, sys; from pathlib import Path
R = Path(os.getcwd()); [sys.path.insert(0, str(R / s)) for s in ("", "media", "core")]
import remix, soundfile as sf
b, _ = sf.read(sys.argv[2], dtype="float32"); print(remix._mix(sys.argv[1], b, os.path.abspath(sys.argv[3])))
PY
