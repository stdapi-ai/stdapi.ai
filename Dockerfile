# Container image for AGPL/Community edition of stdapi.ai
#
# A stock Debian image built the ordinary way: distribution packages, a locked
# virtual environment, the application on top.  It serves the same application
# as the Marketplace image, with the same Python version, the same dependencies
# and the same media stack; what the Marketplace image adds is a hardened
# minimal base with no shell and no package manager, an audio-only ffmpeg built
# from source, pruned wheels and a byte-compiled application tree.

# Pinned to the interpreter version this project requires, on the current Debian
# stable.  The patch level floats so every rebuild picks up Debian's security
# updates, which a digest pin would freeze out.
FROM python:3.14-slim-trixie AS builder

# uv is copied from its own digest-pinned image rather than fetched unpinned
# from PyPI: the hash verification the next stage relies on ("uv export"'s
# requirements.txt carries the lock file's hashes) is only as trustworthy as
# the tool doing the verifying, and an unpinned fetch of that tool would trust
# whatever PyPI serves at build time.
COPY --from=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 /uv /usr/local/bin/uv

# Dependencies are installed into a virtual environment the runtime stage copies
# whole, so uv, pip and their caches stay in this stage.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv

WORKDIR /opt/app

RUN uv venv "${VIRTUAL_ENV}"

# Copied before the application so a source change does not reinstall them.  The
# exported file carries the lock file's hashes, which the install verifies.
COPY pyproject.toml uv.lock ./
RUN uv export --quiet --frozen --no-dev --no-emit-project \
        --extra granian --extra opentelemetry --extra mcp --extra webrtc \
        --format requirements.txt --output-file requirements.txt && \
    uv pip install --quiet --requirement requirements.txt && \
    rm requirements.txt

COPY stdapi ./stdapi

# Fetch what the /docs and /redoc pages load in a browser: Swagger UI and ReDoc,
# pinned to an exact release and each verified against the SHA-256 recorded in
# stdapi/docs_assets/__init__.py, plus their upstream licence texts.  Without
# this the pages load them from a CDN at a floating major tag, which an
# air-gapped deployment cannot reach at all.  A digest mismatch or an
# unreachable publisher fails the build.
# The modes come from the build context, where they may be readable by their
# owner alone; the runtime user is not that owner.
RUN "${VIRTUAL_ENV}/bin/python" -m stdapi.docs_assets && \
    chmod -R a+rX /opt/app /opt/venv

FROM python:3.14-slim-trixie

# - ffmpeg: encodes Polly speech to wav/flac/aac/pcm and normalizes any
#   "audio/*" upload (AMR, AIFF, WMA, AU, ...) to FLAC for Bedrock speech models
# - libmagic1: file type detection for python-magic
# - tzdata: timezone support
RUN apt-get update && \
    apt-get install --no-install-recommends --yes ffmpeg libmagic1 tzdata && \
    rm -rf /var/lib/apt/lists/*

# The same uid/gid as the Marketplace image, so a mounted "~/.aws" and the
# Terraform module's task definition behave identically on both.  The home
# directory is left empty by this build and holds no secret, so it is made
# traversable (0755) rather than left at Debian's default 0700: otherwise a
# container run with "--user <host-uid>:<host-gid>" cannot even resolve a
# path under it, which breaks every documented "-v ~/.aws:/home/nonroot/.aws"
# recipe.
RUN groupadd --gid 65532 nonroot && \
    useradd --uid 65532 --gid 65532 --create-home --shell /usr/sbin/nologin nonroot && \
    chmod 0755 /home/nonroot

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/app /opt/app

# Licences of what the image redistributes: the served application, ffmpeg —
# whose Debian package ships its own copyright file — and the documentation
# pages' browser assets, since Apache-2.0 requires its notice to travel with the
# redistribution.
COPY LICENSE-AGPL /usr/share/licenses/stdapi.ai/LICENSE-AGPL
RUN cp -a /opt/app/stdapi/docs_assets/licenses/. /usr/share/licenses/ && \
    mkdir /usr/share/licenses/ffmpeg && \
    cp /usr/share/doc/ffmpeg/copyright /usr/share/licenses/ffmpeg/ && \
    chmod -R a+rX /usr/share/licenses

# The application is imported from the working directory, not installed into the
# environment: the health probe runs with "-S" and never sees site-packages.
WORKDIR /opt/app
USER nonroot

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
    CMD ["/opt/venv/bin/python", "-S", "-m", "stdapi.healthcheck"]

ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "granian", "stdapi.main:app", "--interface", "asgi", "--loop", "uvloop"]
