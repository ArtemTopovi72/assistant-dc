cd C:\Users\Artem\f5-tts-project
& "C:\Users\Artem\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv_applio
$py="venv_applio\Scripts\python.exe"
& $py -m pip install -U pip uv
& venv_applio\Scripts\uv.exe pip install --python $py -r models_ext\applio\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
& $py -c "import torch;print(torch.__version__,torch.cuda.is_available())" > outputs\applio_pip_done.txt 2>&1
