FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app
RUN chmod +x /app/docker/start_all.sh

EXPOSE 8000

# Default: full stack in one container (API + worker + all web portals) so a
# plain `docker run` of the published image delivers the complete product.
# docker-compose services each override this with their own single-process
# command.
CMD ["/app/docker/start_all.sh"]
