# Container image for AGPL/Community edition of stdapi.ai

FROM python:3-alpine AS ffmpeg-builder

# Build ffmpeg with only the audio codecs this server uses, pinned to the
# version Alpine packages.  The binary encodes Polly output to wav/flac/aac/pcm
# and decodes legacy "audio/*" uploads (AMR, AIFF, WMA, AU, ...) to FLAC for the
# Bedrock speech models.  The distribution package links video codecs, X11 and
# font libraries — none of which are reachable from audio transcoding — and
# costs ~130 MB.
RUN apk add --no-cache build-base nasm curl && \
    apk update >/dev/null && \
    ffmpeg_version="$(apk search -x ffmpeg | head -1 | sed 's/^ffmpeg-//; s/-r[0-9]*$//')" && \
    if [ -z "${ffmpeg_version}" ]; then \
        echo "No ffmpeg version found in the Alpine package index" >&2; exit 1; \
    fi && \
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
        --enable-demuxer=wav,mp3,ogg,flac,aac,pcm_s16le,amr,aiff,asf,au \
        --enable-decoder=pcm_s16le,mp3float,vorbis,flac,aac,amrnb,amrwb,pcm_s16be,pcm_s8,pcm_s24be,pcm_s32be,pcm_f32be,pcm_mulaw,pcm_alaw,wmav1,wmav2,wmapro \
        --enable-encoder=pcm_s16le,flac,aac \
        --enable-muxer=wav,flac,adts,pcm_s16le \
        --enable-filter=aresample,aformat,anull \
        --enable-parser=mpegaudio,flac,aac && \
    make -j"$(nproc)" && make install && \
    # The LGPL text ships beside the binary built from these sources.
    mkdir -p /ffmpeg-out/licenses && \
    find . -maxdepth 1 \( -name "COPYING*" -o -name "LICENSE*" \) \
        -exec cp {} /ffmpeg-out/licenses/ \; && \
    printf 'P:ffmpeg\nV:%s-r0\nA:%s\nT:Custom audio-only LGPL ffmpeg build (Polly output to wav/flac/aac/pcm, legacy audio uploads to flac)\nU:https://ffmpeg.org\nL:LGPL-2.1-or-later\no:ffmpeg\n\n' \
        "${ffmpeg_version}" "$(apk --print-arch)" > /ffmpeg-out/apk-entry

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

# Fetch what the /docs and /redoc pages load in a browser: Swagger UI and ReDoc,
# pinned to an exact release and each verified against the SHA-256 recorded in
# stdapi/docs_assets/__init__.py, plus their upstream licence texts.  Without
# this the pages load them from a CDN at a floating major tag, which an
# air-gapped deployment cannot reach at all.  A digest mismatch or an
# unreachable publisher fails the build.
RUN python -m stdapi.docs_assets

# Single source of truth for the services the server constructs clients for:
# botocore/data is pruned to this list and the smoke test below instantiates
# every entry to catch a miss.  The bidirectional stream clients need no entry:
# they are not botocore clients and carry their own generated service models.
ENV BOTOCORE_SERVICES="bedrock bedrock-agent bedrock-agent-runtime bedrock-runtime comprehend meteringmarketplace polly pricing s3 s3vectors secretsmanager sqs ssm sso sso-oidc sts transcribe translate"

# Optimize Python code
# Can't remove "annotated-doc" .dist-info - needed at runtime
# Can't remove "mcp" .dist-info - fastapi_mcp uses importlib.metadata.version("mcp")
RUN find botocore/data -mindepth 1 -maxdepth 1 -type d \
        | grep -vE "/($(echo "${BOTOCORE_SERVICES}" | tr ' ' '|'))$" \
        | xargs rm -rf && \
    find . -type d -name __pycache__ -a -prune -exec rm -rf {} \; && \
    mv annotated_doc-*.dist-info /tmp/ && \
    mv mcp-*.dist-info /tmp/ && \
    # Keep each package's METADATA and SBOM so scanners inventory the Python
    # deps, and its licence and notice files, which redistribution requires.
    find . -maxdepth 1 -name '*.dist-info' -type d \
        -exec sh -c 'find "$1" -type f ! -name METADATA \
            ! -path "$1/licenses/*" ! -path "$1/sboms/*" \
            ! -name "LICEN[CS]E*" ! -name "NOTICE*" \
            ! -name "COPYING*" ! -name "AUTHORS*" -delete' _ {} \; && \
    rm -rf *.virtualenv _virtualenv.pth _virtualenv.py _stdapi.pth && \
    mv /tmp/annotated_doc-*.dist-info . && \
    mv /tmp/mcp-*.dist-info . && \
    python -m compileall . -q -b -j0 -o2 && \
    find . -name "*.py" -type f -delete && \
    find /opt/app/stdapi -type f -exec chmod 644 {} + && \
    find /opt/app/stdapi -type d -exec chmod 755 {} +

# stdapi.aws_bidi is named on its own so the native HTTP/2 library behind the
# bidirectional stream clients is proven loadable here, wherever the server
# happens to import it from.
RUN AWS_DEFAULT_REGION=eu-west-3 python -c "import stdapi.main, stdapi.aws_bidi" && \
    python -c "from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware" && \
    python -c "import os; from botocore.session import Session; s = Session(); [s.create_client(n, region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x') for n in os.environ['BOTOCORE_SERVICES'].split()]"

FROM python:3-alpine

RUN apk add --no-cache tzdata libmagic && \
    adduser -D -u 1000 nonroot && \
    ls -la /home

COPY --from=ffmpeg-builder /ffmpeg-out/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /opt/app /opt/app

# Licences of what the image redistributes: the served application, ffmpeg, and
# the documentation pages' browser assets (Apache-2.0 requires its notice to
# travel with the redistribution).
COPY LICENSE-AGPL /usr/share/licenses/stdapi.ai/LICENSE-AGPL
COPY --from=ffmpeg-builder /ffmpeg-out/licenses /usr/share/licenses/ffmpeg
COPY --from=builder /opt/app/stdapi/docs_assets/licenses /usr/share/licenses/

# Register the self-built ffmpeg in the APK inventory so scanners see it.
COPY --from=ffmpeg-builder /ffmpeg-out/apk-entry /tmp/ffmpeg-apk-entry
RUN cat /tmp/ffmpeg-apk-entry >> /lib/apk/db/installed && rm /tmp/ffmpeg-apk-entry && \
    chmod -R a+rX /usr/share/licenses

USER nonroot
WORKDIR /opt/app

# Bind address and port stay out of CMD so they remain overridable: granian reads
# its own GRANIAN_* variables, but an argument spelled out in CMD would take
# priority over them.  Set GRANIAN_HOST="::" for a dual-stack socket, needed
# wherever a client may resolve the server to an IPv6 address.
# Default GRANIAN_PORT=8000
EXPOSE 8000
ENV GRANIAN_HOST="0.0.0.0" \
    GRANIAN_LOG_LEVEL="critical" \
    GRANIAN_LOG_ACCESS_ENABLED="false"

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD ["python3", "-S", "-m", "stdapi.healthcheck"]

CMD ["python3", "-m", "granian", "stdapi.main:app", "--interface", "asgi", "--loop", "uvloop"]
