# syntax=docker/dockerfile:1
#
# InsurAutoML image, built from the repository (the build context is the repository
# root):
#
#   docker build -f Dockerfiles/Dockerfile -t insurautoml .
#   docker run --rm -p 8765:8765 --shm-size=2g insurautoml
#
# or with Docker Compose: docker compose -f Dockerfiles/docker-compose.yml up --build
#
# Build arguments:
#   PYTHON_VERSION  Python of the image, default 3.12 (the version tested in CI)
#   EXTRAS          optional dependencies, default "informed" (InformedAutoML, with
#                   the C++ extension); "normal" for the InsurAutoML version only,
#                   "informed,nn" adds torch (large, CUDA wheels)

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------- build
# compiles the C++ extension of the informed pipelines into the InsurAutoML wheel
FROM python:${PYTHON_VERSION}-slim-trixie AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libnanoflann-dev \
    && rm -rf /var/lib/apt/lists/*

# only the files of the package (not outputs/builds of the working tree, whether
# or not the builder applies Dockerfile.dockerignore)
WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY InsurAutoML ./InsurAutoML
RUN INFORMEDAUTOML=1 pip wheel --no-deps --wheel-dir /wheels .

# -------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-trixie

ARG EXTRAS=informed

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # the optimization dashboard listens on all interfaces of the container
    # (published with -p 8765:8765)
    INSURAUTOML_DASHBOARD_HOST=0.0.0.0 \
    INSURAUTOML_DASHBOARD_PORT=8765 \
    MLFLOW_DISABLE_AGENT_HINT=1

# OpenMP runtime of the C++ extension and LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheels /wheels
RUN pip install "$(ls /wheels/*.whl)[${EXTRAS}]" \
    && rm -rf /wheels

# command line interface and example data
WORKDIR /app
COPY main.py ./
COPY example/example_data ./example/example_data

# non-root user
RUN useradd --uid 5678 --create-home appuser \
    && chown -R appuser /app
USER appuser

# optimization dashboard
EXPOSE 8765

CMD ["python", "main.py", "--data_folder", "example/example_data", "--train_data", "heart", "--response", "HeartDisease"]
