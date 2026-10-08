Set-Location (Join-Path $PSScriptRoot '..')
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv venv_ace1
$py="venv_ace1\Scripts\python.exe"
& $py -m pip install -U pip
& $py -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r models_ext\acestep1\requirements.txt
& $py -m pip install --no-deps -e models_ext\acestep1
& $py -c "import torch;print(torch.__version__,torch.cuda.is_available())" > outputs\ace1_pip_done.txt 2>&1
