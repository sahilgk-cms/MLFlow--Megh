FROM python:3.11-slim

WORKDIR /megh-pipeline

# Install system deps (optional but useful for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /megh-pipeline

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen


# Copy rest of code
COPY . .

# Default command (can be overridden by docker-compose)
CMD ["uv", "run",  "python", "main.py"]

