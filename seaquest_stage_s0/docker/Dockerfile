# Seaquest Stage-S0 legacy JAX/EnvPool teacher environment.
#
# The CleanRL Seaquest checkpoints are legacy JAX/Flax models trained under
# EnvPool (Linux-x86_64 only). This image reproduces a compatible runtime.
#
# Declared deviation (documented in teacher/candidate_manifest.json):
#   Candidate A's lock pins jax/jaxlib 0.3.17/0.3.15 (numpy 1.21.6); Candidate B
#   pins jax/jaxlib 0.3.25 (numpy 1.24.2). We use a SINGLE venv pinned to the
#   0.3.25 / numpy 1.24.2 stack to load and run BOTH checkpoints. Justification:
#   (1) identical Network/Actor source across candidates; (2) flax deserialization
#   is pytree-structure based, version-robust; (3) the native-setup difference that
#   materially changes behavior is the EnvPool wrapper config, which IS reproduced
#   exactly per candidate (episodic_life / sticky / noop_max / action space).
FROM python:3.9-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    JAX_PLATFORMS=cpu

# System libs: envpool needs libstdc++/glibc (present); opencv-headless avoids libGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned stack (Candidate B lock; serves both candidates). Versions are exact.
# Legacy jaxlib 0.3.x wheels were purged from PyPI; they remain on Google's
# jax-releases bucket, so jax/jaxlib are installed with that find-links index.
RUN pip install --upgrade "pip==24.0" "setuptools<70" wheel
RUN pip install \
        --find-links https://storage.googleapis.com/jax-releases/jax_releases.html \
        "numpy==1.24.2" \
        "jaxlib==0.3.25" "jax==0.3.25" \
        "flax==0.6.0" "optax==0.1.3" "chex==0.1.5" \
        "gym==0.23.1" \
        "envpool==0.8.4" \
        "opencv-python-headless==4.7.0.68" \
        "scipy==1.10.1" \
        "msgpack" "tensorstore==0.1.28" "rich<12" "PyYAML" "typing_extensions==4.4.0"

WORKDIR /work
CMD ["python", "-c", "import jax,flax,envpool,numpy; print('jax',jax.__version__,'flax',flax.__version__,'envpool',envpool.__version__,'numpy',numpy.__version__); print('devices',jax.devices())"]
