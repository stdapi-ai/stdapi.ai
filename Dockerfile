# Container image for AGPL/Community edition of stdapi.ai

FROM python:3-alpine AS ffmpeg-builder

# Build ffmpeg with only the audio encoders this server uses, pinned to the
# version Alpine packages.  The distribution package links video codecs, X11 and
# font libraries — none of which are reachable from audio transcoding — and
# costs ~130 MB for a binary used solely to encode Polly PCM to wav/flac/aac.
RUN apk add --no-cache build-base nasm curl && \
    apk update >/dev/null && \
    ffmpeg_version="$(apk search -x ffmpeg | head -1 | sed 's/^ffmpeg-//; s/-r[0-9]*$//')" && \
    echo "Building ffmpeg ${ffmpeg_version}" && \
    for url in "https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz" \
               "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${ffmpeg_version}.tar.gz"; do \
        echo "Fetching ${url}" && \
        curl -fsSL --connect-timeout 20 --max-time 900 \
            --retry 10 --retry-delay 10 --retry-all-errors --retry-connrefused \
            "${url}" -o /tmp/ffmpeg.tar && break; \
    done && \
    mkdir -p /tmp/ffmpeg && tar xf /tmp/ffmpeg.tar -C /tmp/ffmpeg --strip-components=1 && \
    cd /tmp/ffmpeg && ./configure --prefix=/ffmpeg-out \
        --disable-everything --disable-doc --disable-network --disable-autodetect \
        --disable-debug --disable-shared --enable-static --enable-small \
        --enable-protocol=pipe,file \
        --enable-demuxer=wav,mp3,ogg,flac,aac,pcm_s16le \
        --enable-decoder=pcm_s16le,mp3float,vorbis,flac,aac \
        --enable-encoder=pcm_s16le,flac,aac \
        --enable-muxer=wav,flac,adts \
        --enable-filter=aresample,aformat,anull \
        --enable-parser=mpegaudio,flac,aac && \
    make -j"$(nproc)" && make install && \
    printf 'P:ffmpeg\nV:%s-r0\nA:x86_64\nT:Custom audio-only LGPL ffmpeg build (Polly PCM to wav/flac/aac)\nU:https://ffmpeg.org\nL:LGPL-2.1-or-later\no:ffmpeg\n\n' \
        "${ffmpeg_version}" > /ffmpeg-out/apk-entry

FROM python:3-alpine AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apk add --no-cache tzdata libmagic && \
    pip install uv -q --root-user-action ignore

WORKDIR /opt
COPY pyproject.toml uv.lock ./
RUN mkdir -p build && \
    uv export --quiet --frozen --output-file requirements.txt --format requirements.txt --no-dev --extra granian --extra opentelemetry --extra mcp && \
    uv pip sync --prefix /opt/build pyproject.toml requirements.txt && \
    mkdir -p app && \
    mv build/lib/python*/site-packages/* app/

WORKDIR /opt/app

COPY stdapi /opt/app/stdapi

# Optimize Python code
# Can't remove "annotated-doc" .dist-info - needed at runtime
# Can't remove "mcp" .dist-info - fastapi_mcp uses importlib.metadata.version("mcp")
# botocore/data is pruned to the services the server constructs clients for;
# the smoke test below instantiates every allowlisted client to catch a miss.
RUN find botocore/data -mindepth 1 -maxdepth 1 -type d \
        | grep -vE '/(bedrock|bedrock-agent|bedrock-agent-runtime|bedrock-runtime|comprehend|meteringmarketplace|polly|pricing|s3|secretsmanager|ssm|sso|sso-oidc|sts|transcribe|translate)$' \
        | xargs rm -rf && \
    find . -type d -name __pycache__ -a -prune -exec rm -rf {} \; && \
    mv annotated_doc-*.dist-info /tmp/ && \
    mv mcp-*.dist-info /tmp/ && \
    # Keep each package's METADATA so scanners inventory the Python deps.
    find . -maxdepth 1 -name '*.dist-info' -type d \
        -exec sh -c 'find "$1" -type f ! -name METADATA -delete' _ {} \; && \
    rm -rf *.virtualenv _virtualenv.pth _virtualenv.py _stdapi.pth && \
    mv /tmp/annotated_doc-*.dist-info . && \
    mv /tmp/mcp-*.dist-info . && \
    python -m compileall . -q -b -j0 -o2 && \
    find . -name "*.py" -type f -delete && \
    find /opt/app/stdapi -type f -exec chmod 644 {} + && \
    find /opt/app/stdapi -type d -exec chmod 755 {} +

RUN AWS_DEFAULT_REGION=eu-west-3 python -c "import stdapi.main" && \
    python -c "from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware" && \
    python -c "from botocore.session import Session; s = Session(); [s.create_client(n, region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x') for n in ('bedrock', 'bedrock-agent', 'bedrock-agent-runtime', 'bedrock-runtime', 'comprehend', 'meteringmarketplace', 'polly', 'pricing', 's3', 'secretsmanager', 'ssm', 'sso', 'sso-oidc', 'sts', 'transcribe', 'translate')]"

FROM python:3-alpine

RUN apk add --no-cache tzdata libmagic && \
    adduser -D -u 1000 nonroot && \
    ls -la /home

COPY --from=ffmpeg-builder /ffmpeg-out/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /opt/app /opt/app

# Register the self-built ffmpeg in the APK inventory so scanners see it.
COPY --from=ffmpeg-builder /ffmpeg-out/apk-entry /tmp/ffmpeg-apk-entry
RUN cat /tmp/ffmpeg-apk-entry >> /lib/apk/db/installed && rm /tmp/ffmpeg-apk-entry

USER nonroot
WORKDIR /opt/app

# Default GRANIAN_PORT=8000
EXPOSE 8000
ENV GRANIAN_LOG_LEVEL="critical" \
    GRANIAN_LOG_ACCESS_ENABLED="false"

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD wget --quiet --tries=1 --spider http://localhost:${GRANIAN_PORT:-8000}/health || exit 1

CMD ["python3", "-m", "granian", "stdapi.main:app", "--host", "0.0.0.0", "--interface", "asgi", "--no-ws", "--loop", "uvloop"]
