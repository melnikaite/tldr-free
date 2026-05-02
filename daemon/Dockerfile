FROM python:3.11-slim

# System deps:
# - ffmpeg: audio extraction / re-encoding (used by yt-dlp postprocessor)
# - ca-certificates: HTTPS
# - curl + unzip: to install deno below
# - deno: a JavaScript runtime that yt-dlp uses to evaluate YouTube's signature
#   code. Without it yt-dlp falls back to the android_vr_player API which
#   often returns a degraded format set (no pure audio streams when the
#   session is cookied), causing "Requested format is not available".
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl unzip \
 && curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
 && apt-get purge -y curl unzip \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only pyproject first for better layer caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Source mounted at runtime via docker-compose volume; keep a baseline copy
# so the image is usable on its own.
COPY src ./src

# Tests are baked into the image so `task test` (pytest) sees them. They are
# also mounted at runtime via docker-compose for hot-reload during dev.
COPY tests ./tests

# Entrypoint refreshes yt-dlp + youtube-transcript-api on every server start.
# See docker-entrypoint.sh for rationale (no rebuild needed for routine updates).
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8765"]
