# Container image of stdapi.ai: a minimal base with no shell and no package
# manager, an audio-only ffmpeg built from source, pruned wheels and a
# byte-compiled application tree.  Build context: this repository's root.

FROM cgr.dev/chainguard/wolfi-base AS system-packages

# Runtime package database, seeded into the staging root below and merged back
# into it so the final image inventory stays accurate.
COPY --from=cgr.dev/chainguard/python:latest /usr/lib/apk/db/installed /tmp/runtime-db

# Install required system packages
# - tzdata: timezone support
# - libmagic: file type detection for python-magic
# Installed into a staging root so the runtime image receives only these
# packages and their dependency closure, not wolfi-base's own busybox and
# apk-tools.  "--initdb" starts from an empty root, hence the explicit keys and
# repositories, and the runtime image's own database is copied in as that root's
# starting state: an empty root makes apk resolve "so:libc.so.6" from scratch,
# and Wolfi publishes several glibc packages at once, so the staged closure
# picks up a second, older libc and "COPY /rootfs/usr/lib" then downgrades the
# runtime's own — leaving every binary built against the newer one unable to
# start.  The two databases are finally concatenated, deduplicated by package
# name, so vulnerability scanners see both the Python runtime and these packages
# instead of one set hiding the other.
RUN apk add --no-cache --initdb --root /rootfs \
        --keys-dir /etc/apk/keys \
        --repositories-file /etc/apk/repositories && \
    cp /tmp/runtime-db /rootfs/usr/lib/apk/db/installed && \
    apk add --no-cache --root /rootfs \
        --keys-dir /etc/apk/keys \
        --repositories-file /etc/apk/repositories \
        tzdata libmagic && \
    awk 'BEGIN{RS="";ORS="\n\n"} {n=split($0,l,"\n"); for(i=1;i<=n;i++) if(l[i]~/^P:/) p=l[i]; if(!(p in seen)){seen[p]=1; print}}' \
        /tmp/runtime-db /rootfs/usr/lib/apk/db/installed > /tmp/merged-db && \
    mv /tmp/merged-db /rootfs/usr/lib/apk/db/installed

# Built on the runtime image's own development variant rather than on a bare
# Wolfi: the two floating tags are rebuilt independently, and a compiler whose
# glibc is one release ahead of the runtime's produces a binary the runtime
# cannot start.
FROM cgr.dev/chainguard/python:latest-dev AS ffmpeg-builder

USER root

# Build ffmpeg with only the audio codecs this server uses, pinned to the
# version Wolfi packages.  The distribution package links video codecs (x264,
# x265, aom, dav1d), X11 and font libraries, plus SDL2 for the ffplay binary —
# none of which are reachable from audio transcoding — and costs ~180 MB for a
# binary used solely to encode Polly speech to wav/flac/aac/pcm and to normalize
# any "audio/*" upload libmagic recognizes (AMR, AIFF, WMA, AU, ...) to FLAC for
# Bedrock speech models.  Enabling no GPL component also keeps the result under
# the LGPL.
RUN apk add --no-cache build-base nasm curl && \
    apk update >/dev/null && \
    # Wolfi names the package after the version it carries ("ffmpeg-9.0"), which
    # has as many components as upstream gave the release — two for 9.0. Only
    # the parenthesized version is read, so a package name never supplies digits.
    ffmpeg_version="$(apk add --simulate ffmpeg 2>&1 \
        | sed -nE 's/.*Installing ffmpeg[^ ]* \(([0-9][0-9.]*)-r[0-9]+\).*/\1/p' \
        | head -1)" && \
    if [ -z "${ffmpeg_version}" ]; then \
        echo "No ffmpeg version found in the Wolfi package index" >&2; exit 1; \
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
    printf 'P:ffmpeg\nV:%s-r0\nA:%s\nT:Custom audio-only LGPL ffmpeg build (Polly speech encode, audio upload decode)\nU:https://ffmpeg.org\nL:LGPL-2.1-or-later\no:ffmpeg\n\n' \
        "${ffmpeg_version}" "$(apk --print-arch)" > /ffmpeg-out/apk-entry

FROM cgr.dev/chainguard/python:latest-dev AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1

USER root

# uv is copied from its own digest-pinned image rather than fetched unpinned
# from PyPI: the hash check of the exported requirements is only as
# trustworthy as the tool doing it.
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /usr/local/bin/uv

# Install minimal dependencies
RUN apk add --no-cache tzdata libmagic

# Install dependencies
WORKDIR /opt
COPY pyproject.toml uv.lock ./
RUN mkdir -p build && \
    uv export --quiet --frozen --output-file requirements.txt --format requirements.txt --no-dev --extra granian --extra opentelemetry --extra mcp --extra webrtc && \
    uv pip sync --prefix /opt/build pyproject.toml requirements.txt && \
    mkdir -p app && \
    mv build/lib/python*/site-packages/* app/

WORKDIR /opt/app

# Install application
COPY stdapi /opt/app/stdapi

# Fetch what the /docs and /redoc pages load in a browser: Swagger UI and ReDoc,
# pinned to an exact release and each verified against the SHA-256 recorded in
# stdapi/docs_assets/__init__.py, plus their upstream licence texts.  Without
# this the pages load them from a CDN at a floating major tag, which an
# air-gapped deployment cannot reach at all.  A digest mismatch or an
# unreachable publisher fails the build.
RUN python -m stdapi.docs_assets

# Optimize Python code
# Can't remove "annotated-doc" .dist-info
# Can't remove "mcp" .dist-info - fastapi_mcp uses importlib.metadata.version("mcp")
# botocore/data is pruned to the services the server constructs clients for;
# the smoke test below instantiates every allowlisted client to catch a miss.
# Declared as a build arg so the prune and that smoke test share one list.
# The bidirectional stream clients need no entry: they are not botocore clients
# and carry their own generated service models.
ARG BOTO_SERVICES="bedrock bedrock-agent bedrock-agent-runtime bedrock-runtime cloudwatch comprehend dynamodb polly pricing s3 s3vectors secretsmanager sqs ssm sso sso-oidc sts transcribe translate"
RUN find botocore/data -mindepth 1 -maxdepth 1 -type d \
        | grep -vE "/($(echo "$BOTO_SERVICES" | tr ' ' '|'))\$" \
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
    # dist-info is excluded: PyAV declares AUTHORS.py as one of its licence
    # files, and a deleted declared licence file is a redistribution gap.
    find . -name "*.py" -type f ! -path "*.dist-info/*" -delete && \
    find /opt/app/stdapi -type f -exec chmod 644 {} + && \
    find /opt/app/stdapi -type d -exec chmod 755 {} +

# Validate minimal import, the proxy-headers middleware dependency, and that
# every pruned-allowlist AWS client still constructs.  stdapi.aws_bidi is named
# on its own so the native HTTP/2 library behind the bidirectional stream
# clients is proven loadable here, wherever the server imports it from, and
# stdapi.realtime_webrtc so the WebRTC media stack (aiortc, av, pylibsrtp)
# loads its native libraries on this base.
RUN AWS_DEFAULT_REGION=eu-west-3 python -c "import stdapi.main, stdapi.aws_bidi, stdapi.realtime_webrtc" && \
    python -c "from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware" && \
    python -c "import os; from botocore.session import Session; s = Session(); [s.create_client(n, region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x') for n in os.environ['BOTO_SERVICES'].split()]"

# Register the self-built ffmpeg in the APK inventory so scanners see it.
COPY --from=system-packages /rootfs/usr/lib/apk/db/installed /tmp/base-db
COPY --from=ffmpeg-builder /ffmpeg-out/apk-entry /tmp/ffmpeg-entry
RUN cat /tmp/base-db /tmp/ffmpeg-entry > /tmp/merged-installed

# Licences of what the image redistributes, staged here because the runtime
# image has no shell to make them readable to its non-root user.  The
# documentation pages' browser assets are covered too: Apache-2.0 requires its
# notice to travel with the redistribution.
COPY LICENSE-AGPL /opt/licenses/stdapi.ai/LICENSE-AGPL
COPY --from=ffmpeg-builder /ffmpeg-out/licenses /opt/licenses/ffmpeg
RUN cp -a /opt/app/stdapi/docs_assets/licenses/. /opt/licenses/ && \
    chmod -R a+rX /opt/licenses

# The home directory holds no secret, so it is made traversable (0755) rather
# than left at the base image's 0700: otherwise a container run with
# "--user <host-uid>:<host-gid>" cannot resolve a path under it, which breaks
# every documented "-v ~/.aws:/home/nonroot/.aws" recipe.
RUN install -d -m 0755 -o 65532 -g 65532 /opt/home/nonroot

FROM cgr.dev/chainguard/python:latest

# The MCP Registry validates this against the name in the server.json it is
# given and rejects a mismatch, so the image has to carry it before the server
# can be published. Declared here rather than at push time so every build of
# this file carries it and the container suite can check it. The namespace is
# claimed on the stdapi.ai apex with the registry's v=MCPv1 TXT record, which
# must be published before the first mcp-publisher publish.
LABEL io.modelcontextprotocol.server.name="ai.stdapi/stdapi-ai"

WORKDIR /opt/app

COPY --from=system-packages /rootfs/usr/lib /usr/lib
COPY --from=builder /tmp/merged-installed /usr/lib/apk/db/installed

COPY --from=ffmpeg-builder /ffmpeg-out/bin/ffmpeg /usr/bin/
COPY --from=system-packages /rootfs/usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=system-packages /rootfs/usr/share/misc/magic.mgc /usr/share/misc/magic.mgc

COPY --from=builder /opt/app .
COPY --from=builder /opt/licenses /usr/share/licenses
COPY --from=builder --chown=65532:65532 /opt/home/ /home/

# Bind address and port stay out of CMD so they remain overridable: granian reads
# its own GRANIAN_* variables, but an argument spelled out in CMD would take
# priority over them.  Set GRANIAN_HOST="::" for a dual-stack socket, needed
# wherever a client may resolve the server to an IPv6 address.
# Default GRANIAN_PORT=8000
VOLUME /tmp
EXPOSE 8000
ENV GRANIAN_HOST="0.0.0.0" \
    GRANIAN_LOG_LEVEL="critical" \
    GRANIAN_LOG_ACCESS_ENABLED="false"

# "-S" skips site: the probe imports nothing outside the standard library.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD ["/usr/bin/python", "-S", "-m", "stdapi.healthcheck"]

CMD ["-m", "granian", "stdapi.main:app", "--interface", "asgi", "--loop", "uvloop"]
