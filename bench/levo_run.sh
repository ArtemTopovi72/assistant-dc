#!/bin/bash
# LeVo 2 on this machine:  bash bench/levo_run.sh in.jsonl out_dir [extra generate.py flags]
ROOT=/c/Users/Artem/f5-tts-project
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
cd $ROOT/models_ext/levo_code
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/"
$ROOT/venv_levo/Scripts/python.exe generate.py --ckpt_path songgeneration_v2_large --input_jsonl "$IN" --save_dir "$OUT" --generate_type mixed --use_flash_attn --low_mem "$@"
