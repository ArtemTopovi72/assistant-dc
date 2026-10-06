# Train an RVC v2 voice in Applio on one vocal stem, then convert a vocal with it.
#   bash bench/rvc_train.sh NAME dataset_dir [epochs=300]
set -e
cd "$(dirname "$0")/../models_ext/applio"
PY=../../venv_applio/Scripts/python.exe; N=$1; D=$(cd "$2" && pwd -W); E=${3:-300}
export PYTHONUTF8=1
$PY core.py prerequisites 2>&1 | tail -2 || true
$PY core.py preprocess --model-name $N --dataset-path "$D" --sample-rate 40000 --cut-preprocess Automatic --process-effects
$PY core.py extract --model-name $N --f0-method rmvpe --sample-rate 40000 --gpu 0
$PY core.py train --model-name $N --sample-rate 40000 --total-epoch $E --save-every-epoch 50 --save-only-latest --batch-size 8 --gpu 0 --pretrained --index-algorithm Auto
$PY core.py index --model-name $N
echo TRAIN_DONE
