# The image run_code executes user scripts in (see code_runner.py).
#
# python:3.12-slim has NOTHING beyond the stdlib, so every collage, chart or
# spreadsheet job started with a minute of `install_packages` -- inside a
# container whose /tmp is 64 MB of RAM. This image ships the libraries such
# jobs actually reach for, so the first script just runs.
#
# Torch is a build option (WITH_TORCH=1): the CPU wheel alone is ~750 MB and
# the container is capped at 2 GB RAM, so it is off by default.
FROM python:3.12-slim

ARG WITH_TORCH=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1

# libgl/libglib: opencv-python-headless still needs them at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        pillow numpy scipy pandas matplotlib \
        opencv-python-headless imageio scikit-image imagehash \
        openpyxl python-docx pypdf python-pptx \
        requests beautifulsoup4 lxml tqdm pyyaml graphviz networkx \
        pytest \
    && if [ "$WITH_TORCH" = "1" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
       fi

# House helpers: `import assistant_tools` (dedupe, sharpness, make_collage, ...).
COPY assistant_tools.py /usr/local/lib/python3.12/site-packages/assistant_tools.py
