# Batch measure service for the AvatarX cardio (AFib) pipeline.
#
#   docker build -t avatarx-cardio-measure .
#   docker run -p 8770:8770 \
#     -e AFIB_ALLOW_ORIGIN=https://staging.example.com \
#     -e AFIB_MAX_COLLAPSED_FRACTION=0.05 \
#     avatarx-cardio-measure
#
# Serves POST /api/process-video and GET /healthz. Stateless: uploads are
# written to a temp dir, analysed, and deleted unless AFIB_KEEP_UPLOADS=1.
#
# PORT: honours $PORT if the host platform sets it (Railway, Render, Cloud
# Run all inject this and route to whatever port the app actually bound —
# falls back to 8770 for a plain `docker run` with no PORT set.
#
# 3.12, not 3.13/3.14 — see requirements-measure.txt.
FROM python:3.12-slim

# ffmpeg is a HARD runtime requirement (downscale + trim); libGL/libglib are
# opencv's shared-library deps even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-measure.txt .
RUN pip install --no-cache-dir -r requirements-measure.txt

# The pipeline packages the service actually imports. research/, training/
# and evaluation/ are deliberately excluded: the measure path never imports
# them, and research/ is import-quarantined from app/ by the repo's own
# transitive audit — leaving it out keeps that true by construction.
COPY app/        app/
COPY inference/  inference/
COPY capture/    capture/
COPY rppg/       rppg/
COPY beats/      beats/
COPY features/   features/
COPY preprocessing/ preprocessing/
COPY heads/      heads/
COPY datasets/   datasets/
COPY configs/    configs/
COPY models/     models/
COPY activity/   activity/
COPY protocol/   protocol/
COPY trend/      trend/
COPY rbcg/       rbcg/
COPY run_measure.py .

# Bind all interfaces inside the container; publish with -p.
ENV AFIB_MAX_CONCURRENT=2 \
    AFIB_MAX_UPLOAD_MB=256 \
    PYTHONUNBUFFERED=1

EXPOSE 8770

# Shell form (not the usual exec-array JSON) so $PORT is substituted by the
# shell at healthcheck time, using whatever the platform actually injected.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT:-8770}/healthz',timeout=4).status==200 else 1)"

# Shell form + `exec` so python replaces the shell as PID 1 and receives
# signals directly (SIGTERM on deploy/restart) rather than the shell eating
# them. ${PORT:-8770}: Railway/Render/Cloud Run set $PORT; a bare
# `docker run` with none set falls back to 8770, matching EXPOSE above.
CMD exec python run_measure.py --host 0.0.0.0 --port ${PORT:-8770}
