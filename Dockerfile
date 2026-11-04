# Base is a build arg so one file serves both worlds:
#   docker compose build                      -> CUDA devel image (compiles llama.cpp for your GPU)
#   docker build --build-arg BASE=python:3.12-slim --build-arg CMAKE_ARGS="" .  -> small CPU image
ARG BASE=python:3.12-slim
FROM ${BASE}

ARG CMAKE_ARGS=""

# python:slim already has python; the nvidia base gets 3.10 from apt. espeak-ng is Kokoro's phonemizer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng \
 && (command -v python3 >/dev/null || apt-get install -y --no-install-recommends python3 python3-pip) \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf "$(command -v python3)" /usr/local/bin/python \
 && ln -sf "$(command -v pip3)" /usr/local/bin/pip

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# llama.cpp: CUDA build compiles from source (10-ish min, arch-pinned by compose);
# CPU build grabs a prebuilt wheel and falls back to a source build if unavailable.
RUN if [ -n "$CMAKE_ARGS" ]; then \
      CMAKE_ARGS="$CMAKE_ARGS" pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python>=0.2.90"; \
    else \
      pip install --no-cache-dir "llama-cpp-python>=0.2.90" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      || pip install --no-cache-dir "llama-cpp-python>=0.2.90"; \
    fi

COPY app/ ./app/
WORKDIR /srv/app

ENV INPUT_DIR=/data/input OUTPUT_DIR=/data/output MODELS_DIR=/data/models
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
