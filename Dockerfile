FROM python:3.11-slim

WORKDIR /megh-pipeline

# Install system deps (optional but useful for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Add venv to PATH
ENV PATH="/megh-pipeline/.venv/bin:$PATH"

# Copy rest of code
COPY . .

# Default command (can be overridden by docker-compose)
CMD ["uv", "run",  "python", "main.py"]

