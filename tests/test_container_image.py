"""Tests running against the container images the project ships.

The runtime images build a minimal ffmpeg (``--disable-everything`` plus
component whitelists), so a codec the server needs can be missing from an image
while every developer machine, with its full distribution ffmpeg, stays green.
These tests close that gap: they build each configured image and then require
its ffmpeg to actually perform every transcode the application can ask for, with
the argument vector production uses. The expectations are derived from the
application's own tables, never transcribed from a Dockerfile, so a format added
to the server fails here until the image can serve it.

The same images also rewrite the application itself: ``botocore/data`` is pruned
to the services the server uses, every ``.py`` is deleted once ``compileall -b``
has produced the ``.pyc`` beside it, most ``.dist-info`` content is stripped, and
the final stage is a filesystem the builder's own smoke checks never saw. The
runtime contracts that survive only if all of that was right — the application
importing at all, every AWS client still constructing, libmagic and its database
answering, the timezone database, the kept metadata, the licence texts the
redistribution of those packages requires, a non-root uid, a health probe that
still passes once ``TRUSTED_HOSTS`` makes the server validate Host headers — are
pinned in :class:`TestFinalImageRuntime`, again against the application's own
code rather than against a Dockerfile.

The images under test come from ``STDAPI_CONTAINER_DOCKERFILES``, a
comma-separated list of ``label=path`` entries (absolute, or relative to the
repository root), defaulting to ``community=Dockerfile``. The build context is
always the repository root. ``STDAPI_CONTAINER_IMAGE_<LABEL>`` supplies an
already-built tag and skips that label's build, and ``CONTAINER_ENGINE``
overrides the engine command (e.g. ``flatpak-spawn --host podman``).

Ref: https://docs.podman.io/en/latest/markdown/podman-build.1.html
     https://ffmpeg.org/ffmpeg-formats.html
     stdapi/media.py:encode_audio_stream
     stdapi/models/audio/amazon_polly.py:AudioModel.tts
     stdapi/models/audio/_default.py:AudioModel._audio_content_block
     stdapi/aws.py:get_client
     stdapi/input_file.py:_magic_detect
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import shlex
import shutil
import struct
import subprocess
import sys
import time
import wave
from array import array
from contextlib import contextmanager
from functools import cache
from importlib.resources import files
from io import BytesIO
from os import O_CREAT, O_EXCL, O_WRONLY, environ, fdopen
from os import open as os_open
from pathlib import Path
from secrets import token_hex
from tempfile import gettempdir
from typing import TYPE_CHECKING, get_args
from urllib.request import urlopen
from zlib import compress, crc32

import pytest
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from stdapi.aws_bedrock import MIME_TYPES_TO_AUDIO_TYPE, MIME_TYPES_TO_DOCUMENT_TYPE
from stdapi.docs_assets import ASSETS_PATH, BROWSER_ASSETS
from stdapi.media import _ffmpeg_args
from stdapi.models.audio.amazon_polly import (
    _FORMAT,
    _FORMAT_ENCODE,
    _OPENAI_PCM_SAMPLE_RATE,
    _POLLY_DEFAULT_PCM_SAMPLE_RATE,
)
from stdapi.types.openai_audio import AudioFileFormat
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from starlette.types import Message, Receive, Scope, Send

#: All tests here need a built image, which takes tens of minutes to produce.
pytestmark = pytest.mark.container

#: Container engine command, split so an indirection like flatpak-spawn works.
_ENGINE = tuple(environ.get("CONTAINER_ENGINE", "podman").split())

#: Images to test, as ``label=dockerfile`` entries; the repository's own image by default.
_DOCKERFILES_SETTING = environ.get(
    "STDAPI_CONTAINER_DOCKERFILES", "community=Dockerfile"
)

#: Mount point of the host root inside a toolbox container.
_HOST_ROOT_PREFIX = "/run/host"

#: Engine commands run outside this process's mount namespace, which resolve host paths.
_HOST_SIDE_ENGINE_MARKERS = ("flatpak-spawn", "--remote")

#: Seconds allowed for one image build; the whole ffmpeg is compiled from source.
_BUILD_TIMEOUT = 3600

#: Seconds allowed for any short-lived engine command.
_RUN_TIMEOUT = 300

#: Seconds allowed for the server to answer its first health probe.
_BOOT_TIMEOUT = 120

#: Seconds between two health probes of the starting server.
_POLL_INTERVAL = 1.0

#: Seconds a single HTTP probe waits for its answer.
_PROBE_TIMEOUT = 5.0

#: Seconds allowed for the strategic test run against the started image.
_SUITE_TIMEOUT = 3600

#: Seconds the server gets to stop cleanly before it is killed.
_STOP_TIMEOUT = "10"

#: Trailing lines of build output shown when an image build fails.
_BUILD_LOG_LINES = 50

#: Trailing characters of captured output attached to a failure message.
_LOG_CHARS = 8192

#: Port the image's server listens on.
_SERVER_PORT = 8000

#: Absolute path of ffmpeg in the images, used as an entrypoint override.
_FFMPEG_BINARY = "/usr/bin/ffmpeg"

#: Region the in-image checks run with; the application refuses to import without one.
_IMAGE_REGION = "eu-west-3"

#: Region named when an AWS client is constructed offline, and never contacted.
_CLIENT_PROBE_REGION = "us-east-1"

#: Application functions whose first positional argument names an AWS service.
_CLIENT_FACTORIES = frozenset(
    {"get_client", "create_client", "call_with_region_failover"}
)

#: Application class whose ``(service, region)`` tuples warm the startup client pool.
_CLIENT_POOL_CONSTRUCTOR = "AWSConnectionManager"

#: Services botocore loads itself to resolve SSO credentials, never named in the code.
_CREDENTIAL_SERVICES = ("sso", "sso-oidc")

#: Distributions whose metadata must survive the ``.dist-info`` stripping.
_METADATA_DISTRIBUTIONS = ("mcp", "annotated-doc", "botocore", "fastapi")

#: Timezone constructed in the image; a regional key needs the database, unlike UTC.
_TIMEZONE_KEY = "Europe/Paris"

#: MIME type libmagic must report for a WAV upload, keyed on by the Bedrock audio map.
_WAV_MIME_TYPE = "audio/x-wav"

#: MIME type libmagic must report for a PNG, which the image and video routes key on.
_PNG_MIME_TYPE = "image/png"

#: MIME type libmagic must report for text, keyed on by the Bedrock document map.
_TEXT_MIME_TYPE = "text/plain"

#: Plain-text buffer handed to libmagic, short enough to stay a single ASCII line.
_TEXT_SAMPLE = b"stdapi.ai container image test\n"

#: Shell the engine runs a ``CMD-SHELL`` healthcheck through.
_HEALTHCHECK_SHELL = "/bin/sh"

#: Healthcheck test vector prefix meaning the image declares no healthcheck.
_NO_HEALTHCHECK = "NONE"

#: Dockerfile instruction the build history records for a declared healthcheck.
_HEALTHCHECK_INSTRUCTION = "HEALTHCHECK "

#: Separator between a healthcheck's options and the command it runs.
_HEALTHCHECK_COMMAND = " CMD "

#: ``TRUSTED_HOSTS`` deployments the container's own health probe must survive.
_TRUSTED_HOSTS_CASES: Mapping[str, tuple[str, ...]] = {
    "unset": (),
    "documented-hosts": ("api.example.com", "www.example.com"),
    "wildcard-subdomain": ("*.example.com",),
    "any-host": ("*",),
}

#: Marker the probe script prefixes the healthcheck's exit status with.
_PROBE_EXIT_MARKER = "HEALTHCHECK_EXIT "

#: Marker the probe script prefixes the request its stub server recorded with.
_PROBE_REQUEST_MARKER = "PROBE_REQUEST "

#: Documentation pages rendered in the image, and the files each one has to load.
_DOCUMENTATION_PAGES = {
    "/docs": ("swagger-ui-bundle.js", "swagger-ui.css"),
    "/redoc": ("redoc.standalone.js",),
}

#: Hosts FastAPI's own pages reach, and which no page the image serves may name.
_THIRD_PARTY_HOSTS = (
    "cdn.jsdelivr.net",
    "fonts.googleapis.com",
    "fastapi.tiangolo.com",
)

#: Directories holding the licence texts of everything the image redistributes.
_LICENSE_DIRECTORIES = (
    "/usr/share/licenses/stdapi.ai",
    "/usr/share/licenses/ffmpeg",
    "/usr/share/licenses/swagger-ui-dist",
    "/usr/share/licenses/redoc",
)

#: Distributions whose METADATA declares licence files, one per wheel layout.
_LICENSED_DISTRIBUTIONS = ("botocore", "fastapi")

#: In-image program importing the served application and reporting its route count.
_IMPORT_PROGRAM = "import stdapi.main; print(len(stdapi.main.app.routes))"

#: In-image program reporting the effective user id the image runs its command as.
_EUID_PROGRAM = "import os; print(os.geteuid())"

#: In-image program constructing one client per service named after the region.
_CLIENT_PROGRAM = """
import sys
from botocore.session import Session

session = Session()
region, *services = sys.argv[1:]
for service in services:
    # Placeholder credentials: construction must not consult the credential
    # chain, and the client is never used to call anything.
    session.create_client(
        service,
        region_name=region,
        aws_access_key_id="probe",
        aws_secret_access_key="probe",
    )
print(len(services))
"""

#: In-image program sniffing each length-prefixed buffer piped to its standard input.
_MAGIC_PROGRAM = """
import json
import struct
import sys

from stdapi.input_file import _magic_detect

data = sys.stdin.buffer.read()
detected = []
offset = 0
while offset < len(data):
    (size,) = struct.unpack_from(">L", data, offset)
    offset += 4
    detected.append(_magic_detect(data[offset : offset + size]))
    offset += size
print(json.dumps(detected))
"""

#: In-image program reporting the size of the icon the documentation pages are branded with.
_FAVICON_PROGRAM = (
    "from importlib.resources import files;"
    "print(len((files('stdapi') / 'favicon.svg').read_bytes()))"
)

#: In-image program reporting the digest of every documentation asset the build fetched.
_DOCS_ASSET_PROGRAM = """
import json

from stdapi.docs_assets import LOCAL_ASSETS, digest

print(json.dumps({n: digest(p.read_bytes()) for n, p in LOCAL_ASSETS.items()}))
"""

#: In-image program rendering a documentation page through the served application.
_DOCS_PAGE_PROGRAM = """
import asyncio
import json
import os
import sys

os.environ["ENABLE_DOCS"] = "true"
os.environ["ENABLE_REDOC"] = "true"

from stdapi.main import app


async def render(path):
    rendered = {"status": 0, "body": bytearray()}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            rendered["status"] = message["status"]
        elif message["type"] == "http.response.body":
            rendered["body"].extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 1234),
        "server": ("localhost", 8000),
    }
    await app(scope, receive, send)
    return {"status": rendered["status"], "body": bytes(rendered["body"]).decode()}


print(json.dumps({path: asyncio.run(render(path)) for path in sys.argv[1:]}))
"""

#: In-image program reporting the module files installed beside the application package.
_LAYOUT_PROGRAM = """
import json
from importlib.util import find_spec
from pathlib import Path

root = Path(next(iter(find_spec("stdapi").submodule_search_locations)))
print(
    json.dumps(
        {
            "root": str(root),
            "sources": sorted(str(path) for path in root.rglob("*.py")),
            "caches": sorted(str(path) for path in root.rglob("__pycache__")),
            "compiled": sum(1 for _ in root.rglob("*.pyc")),
        }
    )
)
"""

#: In-image program reporting the version of each distribution named on its command line.
_METADATA_PROGRAM = """
import json
import sys
from importlib.metadata import version

print(json.dumps({name: version(name) for name in sys.argv[1:]}))
"""

#: In-image program constructing a timezone and counting the ones the database holds.
_TIMEZONE_PROGRAM = """
import json
import sys
from zoneinfo import ZoneInfo, available_timezones

print(json.dumps({"key": ZoneInfo(sys.argv[1]).key, "available": len(available_timezones())}))
"""

#: In-image program resolving each executable named on its command line against PATH.
_WHICH_PROGRAM = """
import json
import sys
from shutil import which

print(json.dumps({name: which(name) or "" for name in sys.argv[1:]}))
"""

#: In-image program answering the probe's ``/health`` request and running the probe.
_STUB_SERVER_PROGRAM = """
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

record = {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.answer(body=True)

    # Starlette serves HEAD on every GET route, so a "--spider" probe works too.
    def do_HEAD(self):
        self.answer(body=False)

    def answer(self, body):
        # Only the probe's own first request: a connection it retries must be
        # served too, but it must not overwrite what is being asserted on.
        record.setdefault("path", self.path)
        record.setdefault("host", self.headers.get("Host", ""))
        record.setdefault("method", self.command)
        payload = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def log_message(self, *args):
        pass


# The same bind as the served command: every IPv4 address, and no IPv6 one.
# Threaded and serving forever, so the probe's own timeout is the only clock in
# play: a single-request handler makes a retried or reopened connection hang
# until the probe gives up, which fails the run for a healthy image.
server = ThreadingHTTPServer(
    ("0.0.0.0", int(os.environ.get("GRANIAN_PORT", "8000"))), Handler
)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
probe = subprocess.run(json.loads(sys.argv[1]), check=False)
print("__EXIT__" + str(probe.returncode))
print("__REQUEST__" + json.dumps(record))
"""

#: In-image program checking every licence file each distribution's METADATA declares.
_LICENSE_PROGRAM = """
import json
from importlib.util import find_spec
from pathlib import Path

root = Path(next(iter(find_spec("stdapi").submodule_search_locations))).parent
declared = {}
missing = []
for info in sorted(root.glob("*.dist-info")):
    metadata = info / "METADATA"
    if not metadata.is_file():
        missing.append(info.name + "/METADATA")
        continue
    names = [
        line.split(":", 1)[1].strip()
        for line in metadata.read_text(errors="replace").splitlines()
        if line.lower().startswith("license-file:")
    ]
    declared[info.name.partition("-")[0].replace("_", "-").lower()] = names
    for name in names:
        # Metadata 2.4 puts the files under "licenses/", older wheels at the root.
        found = next(
            (path for path in (info / "licenses" / name, info / name) if path.is_file()),
            None,
        )
        try:
            content = found.read_bytes() if found is not None else b""
        except OSError:
            content = b""
        if not content:
            missing.append(info.name + "/" + name)
print(json.dumps({"declared": declared, "missing": missing}))
"""

#: In-image program reporting the readable size of every file under each directory.
_LICENSE_DIRECTORY_PROGRAM = """
import json
import sys
from pathlib import Path

report = {}
for name in sys.argv[1:]:
    directory = Path(name)
    files = {}
    for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
        if not path.is_file():
            continue
        try:
            files[str(path.relative_to(directory))] = len(path.read_bytes())
        except OSError:
            files[str(path.relative_to(directory))] = 0
    report[name] = files
print(json.dumps(report))
"""

#: Credential variables the server validates through STS before it serves anything.
_CREDENTIAL_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")

#: AWS variables naming a host file or profile that the container cannot resolve.
_UNFORWARDABLE_AWS_VARS = frozenset(
    {"AWS_PROFILE", "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE"}
)

#: Response formats routed through ffmpeg: what Polly cannot emit, plus resampled pcm.
_FFMPEG_RESPONSE_FORMATS = frozenset(_FORMAT_ENCODE) | {"pcm"}

#: Response formats Polly emits itself, streamed to the caller without re-encoding.
_NATIVE_RESPONSE_FORMATS = frozenset(_FORMAT) | {"mp3"}

#: Frames in a generated audio sample; one second, long enough to survive any encoder.
_SAMPLE_FRAMES = 16000

#: Pitch of the generated audio sample, in Hz; a tone, since encoders may drop silence.
_SAMPLE_TONE_HZ = 440

#: Peak amplitude of the generated audio sample, well under the 16-bit maximum.
_SAMPLE_AMPLITUDE = 8000

#: Sample rate of the handcrafted AIFF sample; a power of two is exact as an 80-bit float.
_AIFF_SAMPLE_RATE = 16384

#: Sample rate of the handcrafted Sun AU sample, the rate mulaw telephony audio uses.
_AU_SAMPLE_RATE = 8000

#: Ratio of output to input bytes when 16 kHz pcm is resampled to OpenAI's 24 kHz.
_RESAMPLE_RATIO = _OPENAI_PCM_SAMPLE_RATE / _POLLY_DEFAULT_PCM_SAMPLE_RATE

#: Tolerance on that ratio; resampling adds a few frames of filter delay.
_RESAMPLE_TOLERANCE = 0.15

#: Legacy upload formats the images enable, which only the transcode fallback accepts.
_LEGACY_INPUTS = ("amr", "aiff16", "aiff24", "asf", "au")

#: Upload formats Bedrock Converse accepts natively, transcoded when a model cannot.
_NATIVE_INPUTS = ("wav", "mp3", "ogg", "flac", "aac")

#: Host ffmpeg arguments producing each sample the stdlib cannot write, keyed by name.
_HOST_SAMPLE_ARGS: Mapping[str, tuple[str, ...]] = {
    "mp3": ("-c:a", "libmp3lame", "-f", "mp3"),
    "ogg": ("-c:a", "libvorbis", "-f", "ogg"),
    "flac": ("-c:a", "flac", "-f", "flac"),
    "aac": ("-c:a", "aac", "-f", "adts"),
    "amr": (
        "-ar",
        "8000",
        "-ac",
        "1",
        "-c:a",
        "libopencore_amrnb",
        "-b:a",
        "12.2k",
        "-f",
        "amr",
    ),
    "asf": ("-c:a", "wmav2", "-b:a", "64k", "-f", "asf"),
}

if shutil.which(_ENGINE[0]) is None:
    pytest.skip(
        f"container engine '{_ENGINE[0]}' is not available", allow_module_level=True
    )


def _parse_dockerfiles(setting: str) -> dict[str, Path]:
    """Parse the ``label=path`` image list into resolved Dockerfile paths.

    Args:
        setting: Value of ``STDAPI_CONTAINER_DOCKERFILES``.

    Returns:
        Each label mapped to its Dockerfile, in the order they were declared.

    Raises:
        pytest.UsageError: If an entry is not a ``label=path`` pair.
    """
    dockerfiles: dict[str, Path] = {}
    for entry in setting.split(","):
        if not entry.strip():
            continue
        label, separator, path = entry.partition("=")
        if not separator or not label.strip() or not path.strip():
            msg = f"STDAPI_CONTAINER_DOCKERFILES entry '{entry}' is not 'label=path'"
            raise pytest.UsageError(msg)
        dockerfiles[label.strip()] = Path(REPO_ROOT, path.strip())
    return dockerfiles


#: Dockerfile of each image under test, keyed by its label.
_DOCKERFILES = _parse_dockerfiles(_DOCKERFILES_SETTING)


def _engine_path(path: Path) -> str:
    """Translate *path* into the namespace of the engine that will resolve it.

    A toolbox container sees the host root under ``/run/host``, which the host's
    own engine does not, so that prefix is stripped for a host-side engine.

    Args:
        path: Path as this test process sees it.

    Returns:
        The path as the engine sees it.
    """
    host_side = any(
        marker in argument
        for argument in _ENGINE
        for marker in _HOST_SIDE_ENGINE_MARKERS
    )
    if host_side and path.is_relative_to(_HOST_ROOT_PREFIX):
        return f"/{path.relative_to(_HOST_ROOT_PREFIX)}"
    return str(path)


def _engine_run(
    args: Sequence[str], *, stdin: bytes = b"", timeout: float = _RUN_TIMEOUT
) -> subprocess.CompletedProcess[bytes]:
    """Run one container-engine command to completion.

    Args:
        args: Engine arguments, without the engine command itself.
        stdin: Bytes piped to the command's standard input.
        timeout: Seconds before the command is killed.

    Returns:
        The completed process, with output captured and its status unchecked.
    """
    return subprocess.run(  # noqa: S603
        [*_ENGINE, *args],
        input=stdin,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


@contextmanager
def _env_file(variables: Mapping[str, str]) -> Iterator[str]:
    """Write *variables* to a private file the engine itself reads.

    A host-side engine does not carry this process's environment across the
    sandbox boundary, so variables merely named on the command line arrive
    unset and the server exits on its first configuration check. An env file
    crosses that boundary, keeps the values off every command line, and is
    created owner-read-write in one atomic step so no one else can read it.

    Args:
        variables: Environment entries for the container.

    Yields:
        Path of the env file, as the engine resolves it.
    """
    path = Path(gettempdir(), f"stdapi-image-test-{token_hex(8)}.env")
    # A value spanning several lines cannot be represented, and the extra lines
    # would be read as entries of their own.
    body = "".join(
        f"{name}={value}\n"
        for name, value in variables.items()
        if "\n" not in value and "\r" not in value
    )
    descriptor = os_open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
    try:
        with fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(body)
        yield _engine_path(path)
    finally:
        path.unlink(missing_ok=True)


def _tail(output: bytes, chars: int = _LOG_CHARS) -> str:
    """Return the decoded trailing *chars* of captured output, for a failure message."""
    return output.decode(errors="replace")[-chars:] or "<no output>"


def _container_logs(container: str) -> str:
    """Return the container's trailing output, for a failure message."""
    result = _engine_run(["logs", container])
    return _tail(result.stdout + result.stderr)


def _container_running(container: str) -> bool:
    """Return whether the container is still running."""
    result = _engine_run(["inspect", "-f", "{{.State.Running}}", container])
    return result.stdout.decode(errors="replace").strip() == "true"


def _pcm_sample(frames: int = _SAMPLE_FRAMES, sample_rate: int = 16000) -> bytes:
    """Return mono 16-bit little-endian PCM holding a tone.

    A tone rather than silence: some encoders emit almost nothing for digital
    silence, which would make a size assertion meaningless.

    Args:
        frames: Number of sample frames.
        sample_rate: Sample rate in Hz, which sets the tone's period.

    Returns:
        The raw PCM bytes.
    """
    samples = array(
        "h",
        (
            int(
                _SAMPLE_AMPLITUDE
                * math.sin(2 * math.pi * _SAMPLE_TONE_HZ * i / sample_rate)
            )
            for i in range(frames)
        ),
    )
    if sys.byteorder != "little":  # pragma: no cover
        samples.byteswap()
    return samples.tobytes()


def _wav_sample() -> bytes:
    """Return the tone as a mono 16 kHz WAV file, written by the standard library."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_POLLY_DEFAULT_PCM_SAMPLE_RATE)
        writer.writeframes(_pcm_sample())
    return buffer.getvalue()


def _aiff_sample(sample_size: int) -> bytes:
    """Return a mono AIFF file of the tone, big-endian, at 16384 Hz.

    AIFF is outside the Bedrock Converse format enum, so it can only reach a
    model through the transcode fallback. The standard library dropped its AIFF
    writer in Python 3.13, hence the handcrafted chunks; 16384 Hz keeps the
    80-bit extended float of the COMM chunk a literal constant.

    Args:
        sample_size: Bits per sample, 16 or 24.

    Returns:
        The AIFF file bytes.
    """
    samples = _pcm_sample(sample_rate=_AIFF_SAMPLE_RATE)
    frames = len(samples) // 2
    if sample_size == 16:
        # Little-endian pairs to big-endian pairs.
        data = b"".join(
            samples[i + 1 : i + 2] + samples[i : i + 1]
            for i in range(0, len(samples), 2)
        )
    else:
        # 24-bit big-endian: the 16-bit value in the high bytes, zero in the low one.
        data = b"".join(
            samples[i + 1 : i + 2] + samples[i : i + 1] + b"\x00"
            for i in range(0, len(samples), 2)
        )
    # 16384 Hz as an 80-bit extended float: exponent 0x400d, mantissa 0x80 then zeros.
    comm = struct.pack(">hLh", 1, frames, sample_size) + b"\x40\x0d\x80" + b"\x00" * 7
    ssnd = struct.pack(">LL", 0, 0) + data
    body = b"AIFF"
    body += b"COMM" + struct.pack(">L", len(comm)) + comm
    body += b"SSND" + struct.pack(">L", len(ssnd)) + ssnd
    return b"FORM" + struct.pack(">L", len(body)) + body


def _au_sample() -> bytes:
    """Return a mono 8-bit mulaw Sun AU file of the tone.

    Sun AU is another legacy upload format outside the Converse enum, and the
    only one exercising the mulaw decoder.

    Returns:
        The AU file bytes.
    """
    samples = array("h", _pcm_sample(sample_rate=_AU_SAMPLE_RATE))
    encoded = bytes(_linear_to_mulaw(sample) for sample in samples)
    # Magic, header size, data size, encoding 1 (8-bit mulaw), sample rate, channels.
    header = struct.pack(">4sLLLLL", b".snd", 24, len(encoded), 1, _AU_SAMPLE_RATE, 1)
    return header + encoded


def _linear_to_mulaw(sample: int) -> int:
    """Encode one 16-bit linear sample as an 8-bit mu-law byte.

    Args:
        sample: Signed 16-bit linear sample.

    Returns:
        The mu-law byte, as the G.711 companding law defines it.
    """
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample) + 132, 32635)
    exponent = magnitude.bit_length() - 8
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _png_sample() -> bytes:
    """Return a 1x1 greyscale PNG, assembled here so no encoder is needed.

    A real file rather than the eight-byte signature alone: libmagic reads on
    past it, and a truncated stream would prove less than the upload path does.

    Returns:
        The PNG file bytes.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        """Return one length-prefixed, CRC-suffixed PNG chunk."""
        return (
            struct.pack(">L", len(payload))
            + kind
            + payload
            + struct.pack(">L", crc32(kind + payload))
        )

    # Width, height, bit depth, colour type 0 (greyscale), then the three
    # compression, filter and interlace bytes the format fixes at zero.
    header = struct.pack(">LLBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        # One scanline: the filter byte, then the single pixel.
        + chunk(b"IDAT", compress(b"\x00\x00"))
        + chunk(b"IEND", b"")
    )


@cache
def _host_sample(name: str) -> bytes | None:
    """Return a sample the host's ffmpeg encodes, or None when it cannot.

    The formats below need an encoder the standard library has none of, and the
    host build may not have one either; the case that needs the sample is
    skipped rather than failed when it is missing.

    Args:
        name: Key into :data:`_HOST_SAMPLE_ARGS`.

    Returns:
        The encoded sample, or None when the host cannot produce it.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    result = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-hide_banner",
            "-f",
            "s16le",
            "-ar",
            str(_POLLY_DEFAULT_PCM_SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *_HOST_SAMPLE_ARGS[name],
            "pipe:1",
        ],
        input=_pcm_sample(),
        capture_output=True,
        check=False,
        timeout=_RUN_TIMEOUT,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _input_sample(name: str) -> bytes:
    """Return the upload sample named *name*, skipping the test when unavailable.

    Args:
        name: One of :data:`_LEGACY_INPUTS` or :data:`_NATIVE_INPUTS`.

    Returns:
        The sample file bytes.
    """
    if name == "wav":
        return _wav_sample()
    if name == "aiff16":
        return _aiff_sample(16)
    if name == "aiff24":
        return _aiff_sample(24)
    if name == "au":
        return _au_sample()
    sample = _host_sample(name)
    if sample is None:
        pytest.skip(f"the host's ffmpeg cannot produce a '{name}' sample")
    return sample


def _image_ffmpeg(image: str, args: Sequence[str], stdin: bytes) -> bytes:
    """Run the image's ffmpeg on *stdin* and return what it wrote to stdout.

    The binary is named as the entrypoint rather than as the command: an image
    declaring its own entrypoint would otherwise append these arguments to it
    and run the server's interpreter on a file called "ffmpeg".

    Args:
        image: Container image tag.
        args: The ffmpeg argument vector, as :func:`stdapi.media._ffmpeg_args`
            builds it for production, starting with the binary name.
        stdin: Bytes piped to ffmpeg's standard input.

    Returns:
        ffmpeg's standard output.
    """
    result = _engine_run(
        [
            "run",
            "--rm",
            "-i",
            "--network=none",
            "--entrypoint",
            _FFMPEG_BINARY,
            image,
            *args[1:],
        ],
        stdin=stdin,
    )
    assert result.returncode == 0, (
        f"'{' '.join(args)}' failed in the image with exit code "
        f"{result.returncode}:\n{_tail(result.stderr)}"
    )
    return result.stdout


@cache
def _inspect_image(image: str) -> dict[str, object]:
    """Return what the engine reports about *image*.

    Args:
        image: Container image tag.

    Returns:
        The first (and only) inspection record.
    """
    result = _engine_run(["image", "inspect", image])
    assert result.returncode == 0, (
        f"inspecting the image '{image}' failed:\n{_tail(result.stderr)}"
    )
    inspected = json.loads(result.stdout)
    assert isinstance(inspected, list), (
        f"the engine's inspection of '{image}' is not a list"
    )
    assert inspected, f"the engine reported nothing for the image '{image}'"
    record: dict[str, object] = inspected[0]
    return record


def _image_config(image: str) -> dict[str, object]:
    """Return the image's configuration block, empty when it declares none.

    Args:
        image: Container image tag.

    Returns:
        The ``Config`` mapping of the inspection record.
    """
    config = _inspect_image(image).get("Config")
    return config if isinstance(config, dict) else {}


def _image_interpreter(image: str) -> str:
    """Return the Python command the image itself runs the server with.

    The two images disagree: one declares only a command starting with the
    interpreter, the other an entrypoint that is the interpreter, so the binary
    is read from whichever the image sets instead of being assumed.

    Args:
        image: Container image tag.

    Returns:
        The interpreter, as an entrypoint override.
    """
    for key in ("Entrypoint", "Cmd"):
        command = _image_config(image).get(key)
        if isinstance(command, list) and command:
            return str(command[0])
    pytest.fail(f"the image '{image}' declares neither an entrypoint nor a command")


def _image_python(image: str, program: str, *arguments: str, stdin: bytes = b"") -> str:
    """Run *program* with the image's own interpreter and return its output.

    The interpreter is named as the entrypoint, for the same reason ffmpeg is:
    an image declaring one would otherwise append these arguments to it. The
    container gets no network and no credentials, only the region the
    application's configuration requires to import.

    Args:
        image: Container image tag.
        program: Python source, run with ``-c``.
        *arguments: Command-line arguments the program reads from ``sys.argv``.
        stdin: Bytes piped to the program's standard input.

    Returns:
        The program's decoded standard output.
    """
    result = _engine_run(
        [
            "run",
            "--rm",
            "-i",
            "--network=none",
            "--env",
            f"AWS_DEFAULT_REGION={_IMAGE_REGION}",
            "--entrypoint",
            _image_interpreter(image),
            image,
            "-c",
            program,
            *arguments,
        ],
        stdin=stdin,
    )
    assert result.returncode == 0, (
        f"the in-image program exited with code {result.returncode}:"
        f"\n{_tail(result.stderr)}"
    )
    return result.stdout.decode(errors="replace")


def _image_healthcheck(image: str) -> tuple[str, ...]:
    """Return the healthcheck test vector *image* declares, empty when it has none.

    The configured field is read first, then the build history: the OCI image
    format has no healthcheck field, so an engine defaulting to it keeps the
    declaration only as the history entry of the instruction, and the probe that
    a Docker-format build of the same Dockerfile would run is invisible in the
    configuration.

    Args:
        image: Container image tag.

    Returns:
        The ``Test`` vector, or an empty tuple when no healthcheck is declared.
    """
    declared = _inspect_image(image).get("Healthcheck") or _image_config(image).get(
        "Healthcheck"
    )
    test = declared.get("Test") if isinstance(declared, dict) else None
    if isinstance(test, list) and test:
        vector = tuple(str(item) for item in test)
        return () if vector[0] == _NO_HEALTHCHECK else vector
    return _history_healthcheck(image)


def _history_healthcheck(image: str) -> tuple[str, ...]:
    """Return the healthcheck the image's build history records, in ``Test`` form.

    Args:
        image: Container image tag.

    Returns:
        The reconstructed ``Test`` vector, empty when the history declares none.
    """
    history = _inspect_image(image).get("History")
    if not isinstance(history, list):
        return ()
    for entry in reversed(history):
        step = entry.get("created_by", "") if isinstance(entry, dict) else ""
        marker = str(step).find(_HEALTHCHECK_INSTRUCTION)
        if marker < 0:
            continue
        # "HEALTHCHECK NONE", and any form without the keyword, leaves nothing
        # to run and is not a declaration this can check.
        _, separator, command = str(step)[marker:].partition(_HEALTHCHECK_COMMAND)
        if not separator:
            continue
        command = command.strip()
        if command.startswith("["):
            return ("CMD", *(str(item) for item in json.loads(command)))
        return ("CMD-SHELL", command)
    return ()


def _healthcheck_binaries(test: Sequence[str]) -> tuple[str, ...]:
    """Return the executables the declared healthcheck vector needs.

    Args:
        test: The healthcheck ``Test`` vector, without its ``NONE`` form.

    Returns:
        Every binary the engine must find in the image to run the probe.
    """
    kind, *rest = test
    if kind == "CMD-SHELL":
        # The engine runs the string through a shell, which must exist too.
        return (_HEALTHCHECK_SHELL, *shlex.split(rest[0])[:1]) if rest else ()
    if kind == "CMD":
        return tuple(rest[:1])
    # The legacy form is the command itself, with no keyword in front.
    return (kind,)


def _healthcheck_argv(test: Sequence[str]) -> list[str]:
    """Return the declared healthcheck vector as the argument list to execute.

    Args:
        test: The healthcheck ``Test`` vector, without its ``NONE`` form.

    Returns:
        The argument list the engine would run, empty when the vector holds none.
    """
    kind, *rest = test
    if kind == "CMD-SHELL":
        # The engine hands the string to a shell, so the probe needs one too.
        return [_HEALTHCHECK_SHELL, "-c", rest[0]] if rest else []
    if kind == "CMD":
        return list(rest)
    # The legacy form is the command itself, with no keyword in front.
    return list(test)


def _marked_line(output: str, marker: str) -> str:
    """Return what the probe script printed after *marker*, empty when absent.

    Args:
        output: The script's standard output.
        marker: Prefix of the line to read.

    Returns:
        The rest of that line, stripped.
    """
    for line in output.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return ""


def _healthcheck_probe(image: str, trusted_hosts: Sequence[str]) -> dict[str, object]:
    """Run the image's own health probe against a stub server inside the image.

    The command comes from the image's declaration rather than from a
    transcription of it, and runs in the environment a deployment would give it,
    so what the stub records is what the real server would have received. A stub
    rather than the server itself: the probe's argument here is the ``Host``
    header it announces, which needs no AWS credentials to observe.

    Args:
        image: Container image tag.
        trusted_hosts: ``TRUSTED_HOSTS`` entries, empty to leave the setting unset.

    Returns:
        The probe's exit status under ``exit``, and the request path and ``Host``
        header the stub recorded under ``path`` and ``host``.
    """
    probe = _healthcheck_argv(_image_healthcheck(image))
    program = _STUB_SERVER_PROGRAM.replace("__EXIT__", _PROBE_EXIT_MARKER).replace(
        "__REQUEST__", _PROBE_REQUEST_MARKER
    )
    environment = (
        ["--env", f"TRUSTED_HOSTS={json.dumps(list(trusted_hosts))}"]
        if trusted_hosts
        else []
    )
    result = _engine_run(
        [
            "run",
            "--rm",
            "--network=none",
            *environment,
            # Driven by the image's own interpreter: the runtime images ship
            # Python, and the hardened one ships no shell at all.
            "--entrypoint",
            probe[0],
            image,
            "-c",
            program,
            json.dumps(probe),
        ]
    )
    output = result.stdout.decode(errors="replace")
    status = _marked_line(output, _PROBE_EXIT_MARKER)
    assert status.isdigit(), (
        f"the probe reported no exit status:\n{output}\n{_tail(result.stderr)}"
    )
    recorded = _marked_line(output, _PROBE_REQUEST_MARKER)
    request: dict[str, object] = json.loads(recorded) if recorded else {}
    return {"exit": int(status), "output": output, **request}


async def _serve_ok(_scope: Scope, _receive: Receive, send: Send) -> None:
    """Answer 200 to any request, standing in for the served application."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _host_is_trusted(host: str, allowed_hosts: Sequence[str]) -> bool:
    """Return whether the middleware the server installs accepts *host*.

    The verdict comes from the very middleware ``stdapi.main`` adds when
    ``TRUSTED_HOSTS`` is set, driven with the request the image's probe made, so
    no matching rule of it is transcribed here.

    Args:
        host: The ``Host`` header the probe announced.
        allowed_hosts: The configured ``TRUSTED_HOSTS`` entries.

    Returns:
        Whether the request reached the application instead of a 400.
    """
    statuses: list[int] = []

    async def receive() -> Message:
        """Return an empty request body, which no host check ever reads."""
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        """Record the status of the response the middleware produced."""
        if message["type"] == "http.response.start":
            statuses.append(int(message["status"]))

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", host.encode())],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", _SERVER_PORT),
    }
    middleware = TrustedHostMiddleware(_serve_ok, allowed_hosts=list(allowed_hosts))

    asyncio.run(middleware(scope, receive, send))

    return statuses == [200]


@cache
def _application_aws_services() -> tuple[str, ...]:
    """Return every AWS service the application constructs a client for.

    The names are parsed out of the application's own source rather than copied
    from a Dockerfile, so a service the server starts using is required of the
    image from the moment it is used. Two shapes are collected: the first
    positional argument of the calls in :data:`_CLIENT_FACTORIES`, and the head
    of every ``(service, region)`` tuple handed to the startup pool in
    :data:`_CLIENT_POOL_CONSTRUCTOR`. Pool keys carry a suffix the client does
    not (``s3.accelerate``, ``<service>.no-retry``), which is dropped here as
    the application drops it before creating the client. The SSO pair is added
    explicitly: botocore, not the application, names those services when it
    resolves credentials from an SSO profile, so no call site mentions them.

    Returns:
        The service names, sorted and deduplicated.
    """
    services: set[str] = set()
    for module in sorted(Path(REPO_ROOT, "stdapi").rglob("*.py")):
        for node in ast.walk(ast.parse(module.read_bytes())):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute):
                name = function.attr
            elif isinstance(function, ast.Name):
                name = function.id
            else:
                continue
            if name in _CLIENT_FACTORIES:
                candidates: list[ast.expr] = node.args[:1]
            elif name == _CLIENT_POOL_CONSTRUCTOR:
                candidates = [
                    spec.elts[0]
                    for spec in ast.walk(node)
                    if isinstance(spec, ast.Tuple) and spec.elts
                ]
            else:
                continue
            services.update(
                candidate.value.split(".", 1)[0]
                for candidate in candidates
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            )
    return tuple(sorted(services.union(_CREDENTIAL_SERVICES)))


def _assert_signature(output_format: str, data: bytes) -> None:
    """Assert *data* carries the container signature its format must have.

    Args:
        output_format: The response format the encode targeted.
        data: What the image's ffmpeg produced.
    """
    assert data, f"the image's ffmpeg produced no '{output_format}' output"
    if output_format == "wav":
        assert data[:4] == b"RIFF", "not a RIFF container"
        assert data[8:12] == b"WAVE", "not a WAVE payload"
    elif output_format == "flac":
        assert data[:4] == b"fLaC", "not a FLAC stream"
    elif output_format == "aac":
        # ADTS frames start with a 12-bit sync word.
        assert data[0] == 0xFF, "not an ADTS frame"
        assert data[1] & 0xF0 == 0xF0, "not an ADTS frame"
    elif output_format == "pcm":
        assert len(data) % 2 == 0, "not whole 16-bit samples"
    else:  # pragma: no cover
        pytest.fail(f"no output signature is known for '{output_format}'; add one")


def _aws_credentials() -> dict[str, str] | None:
    """Return AWS credentials from the environment, else from the AWS CLI.

    The server validates its credentials with STS at startup and exits when they
    are missing, so the boot test cannot run without them. Values are returned
    for the subprocess environment only; they are never put on a command line.

    Returns:
        The credential variables, or None when none could be resolved.
    """
    resolved = {key: environ[key] for key in _CREDENTIAL_VARS if key in environ}
    if "AWS_ACCESS_KEY_ID" in resolved and "AWS_SECRET_ACCESS_KEY" in resolved:
        return resolved
    aws = shutil.which("aws")
    if aws is None:
        return None
    exported = subprocess.run(  # noqa: S603
        [aws, "configure", "export-credentials", "--format", "env"],
        capture_output=True,
        check=False,
        timeout=_RUN_TIMEOUT,
    )
    if exported.returncode != 0:
        return None
    credentials: dict[str, str] = {}
    for line in exported.stdout.decode(errors="replace").splitlines():
        if not line.startswith("export "):
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        if key in _CREDENTIAL_VARS:
            credentials[key] = value.strip("'\"")
    return credentials or None


def _forwarded_settings() -> dict[str, str]:
    """Return the ambient ``AWS_*`` configuration the served tests need.

    The gateway reads its settings from ``AWS_*`` variables, case-insensitively,
    so the container is given the ones this session runs with. The three that
    name a host profile or file are left out: the container cannot resolve them,
    and keeping them would shadow the credentials passed alongside.

    Returns:
        The forwardable variables, with their values.
    """
    return {
        key: value
        for key, value in environ.items()
        if key.upper().startswith("AWS_") and key.upper() not in _UNFORWARDABLE_AWS_VARS
    }


def _probe(port: int, path: str) -> int | None:
    """Return the status of a GET on the started server, or None when unreachable.

    Args:
        port: Published port on the loopback interface.
        path: Path to request.

    Returns:
        The HTTP status code, or None when the request did not complete.
    """
    try:
        with urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=_PROBE_TIMEOUT
        ) as response:
            return int(response.status)
    except OSError:
        return None


def _published_port(container: str) -> int:
    """Return the loopback port the engine published for the server.

    Args:
        container: Container name.

    Returns:
        The published port number.
    """
    result = _engine_run(["port", container, f"{_SERVER_PORT}/tcp"])
    published = result.stdout.decode(errors="replace").strip().splitlines()
    assert published, (
        f"the engine published no port for {_SERVER_PORT}/tcp:"
        f"\n{_tail(result.stderr)}\n{_container_logs(container)}"
    )
    port = published[0].rsplit(":", 1)[-1]
    assert port.isdigit(), f"unexpected port mapping '{published[0]}'"
    return int(port)


def _wait_for_health(container: str, port: int) -> None:
    """Block until the server answers ``GET /health``, or fail with its logs.

    Args:
        container: Container name, inspected so a startup crash fails at once.
        port: Published port on the loopback interface.
    """
    deadline = time.monotonic() + _BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if not _container_running(container):
            pytest.fail(
                "the container exited before answering /health:\n"
                f"{_container_logs(container)}"
            )
        if _probe(port, "/health") == 200:
            return
        time.sleep(_POLL_INTERVAL)
    pytest.fail(
        f"the server did not answer /health within {_BOOT_TIMEOUT}s:\n"
        f"{_container_logs(container)}"
    )


@pytest.fixture(scope="session", params=list(_DOCKERFILES), ids=list(_DOCKERFILES))
def image(request: pytest.FixtureRequest) -> str:
    """Build the image under test, or return the prebuilt tag configured for it.

    Building compiles ffmpeg from source, so ``STDAPI_CONTAINER_IMAGE_<LABEL>``
    short-circuits it with an already-built tag while iterating.

    Args:
        request: Fixture request carrying the image label.

    Returns:
        The image tag to test.
    """
    label = str(request.param)
    prebuilt = environ.get(f"STDAPI_CONTAINER_IMAGE_{label.upper()}")
    if prebuilt:
        return prebuilt
    dockerfile = _DOCKERFILES[label]
    assert dockerfile.is_file(), (
        f"the '{label}' image declares a Dockerfile that does not exist: {dockerfile}"
    )
    tag = f"localhost/stdapi-image-test-{label}:latest"

    built = _engine_run(
        [
            "build",
            "--tag",
            tag,
            "--file",
            _engine_path(dockerfile),
            _engine_path(REPO_ROOT),
        ],
        timeout=_BUILD_TIMEOUT,
    )

    if built.returncode != 0:
        output = (built.stdout + built.stderr).decode(errors="replace").splitlines()
        pytest.fail(
            f"building the '{label}' image failed:\n"
            + "\n".join(output[-_BUILD_LOG_LINES:])
        )
    return tag


class TestPollyEncodeContract:
    """The image encodes every speech response format Polly cannot emit itself.

    Polly synthesizes mp3, Ogg Vorbis, Ogg Opus and pcm; anything else is
    encoded from its pcm output by ffmpeg, and pcm itself is resampled unless
    the caller pinned Polly's rate. Each case below drives the image's ffmpeg
    with the exact argument vector ``stdapi.media`` builds, so a missing muxer,
    encoder or resampler fails here instead of in production.

    Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/models/audio/amazon_polly.py:AudioModel.tts
    """

    @pytest.mark.parametrize("resp_format", sorted(_FORMAT_ENCODE))
    def test_polly_pcm_is_encoded_to_the_response_format(
        self, image: str, resp_format: str
    ) -> None:
        """Each ``_FORMAT_ENCODE`` format is produced from Polly's 16 kHz pcm.

        The cases are the application's own set, so a format added to the
        server fails here until the image ships the codec it needs.

        Ref: stdapi/media.py:_ffmpeg_args
             stdapi/models/audio/amazon_polly.py:_FORMAT_ENCODE
        """
        args = _ffmpeg_args(
            resp_format, "s16le", _POLLY_DEFAULT_PCM_SAMPLE_RATE, 1, None
        )

        output = _image_ffmpeg(image, args, _pcm_sample())

        _assert_signature(resp_format, output)

    def test_pcm_is_resampled_to_the_openai_rate(self, image: str) -> None:
        """Default pcm output is resampled from Polly's 16 kHz to OpenAI's 24 kHz.

        This is the whole ``pcm`` response format in the image: raw signed
        16-bit samples with no container, which needs the ``s16le`` demuxer,
        muxer and resampler all present. The body must carry 1.5x the bytes of
        its input, which is the contract callers decode against.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/models/audio/amazon_polly.py:_OPENAI_PCM_SAMPLE_RATE
        """
        source = _pcm_sample()
        args = _ffmpeg_args(
            "pcm", "s16le", _POLLY_DEFAULT_PCM_SAMPLE_RATE, 1, _OPENAI_PCM_SAMPLE_RATE
        )

        output = _image_ffmpeg(image, args, source)

        _assert_signature("pcm", output)
        assert len(output) / len(source) == pytest.approx(
            _RESAMPLE_RATIO, rel=_RESAMPLE_TOLERANCE
        )

    def test_every_response_format_is_covered(self) -> None:
        """Every documented ``response_format`` is either native or tested here.

        A format added to ``AudioFileFormat`` is served by Polly directly or by
        ffmpeg; the second kind needs image support, so this fails until the new
        format is classified and, when encoded, covered by the cases above.

        Ref: stdapi/types/openai_audio.py:AudioFileFormat
        """
        documented = frozenset(get_args(AudioFileFormat))

        assert documented == _FFMPEG_RESPONSE_FORMATS | _NATIVE_RESPONSE_FORMATS
        assert not _FFMPEG_RESPONSE_FORMATS & _NATIVE_RESPONSE_FORMATS


class TestTranscodeFallbackContract:
    """The image transcodes every audio upload the server accepts into FLAC.

    An upload whose format is outside the Bedrock Converse enum is normalized to
    FLAC before it reaches a model, with ffmpeg probing the format from the pipe
    exactly as it does here. The legacy cases are the reason the images enable
    demuxers and decoders no other route needs.

    Ref: stdapi/models/audio/_default.py:CONVERSE_AUDIO_FORMATS
         stdapi/media.py:encode_audio_stream
    """

    @pytest.mark.parametrize("sample_format", _LEGACY_INPUTS)
    def test_legacy_upload_is_transcoded_to_flac(
        self, image: str, sample_format: str
    ) -> None:
        """AMR-NB, AIFF (16- and 24-bit), WMA and Sun AU uploads decode to FLAC.

        None of these is a Converse audio format, so each one reaches a model
        only through the fallback; each also needs its own demuxer and decoder,
        which a minimal ffmpeg build has no reason to include by default.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_AudioBlock.html
             stdapi/models/audio/_default.py:AudioModel._audio_content_block
        """
        sample = _input_sample(sample_format)

        output = _image_ffmpeg(
            image, _ffmpeg_args("flac", None, None, None, None), sample
        )

        _assert_signature("flac", output)

    @pytest.mark.parametrize("sample_format", _NATIVE_INPUTS)
    def test_common_upload_is_transcoded_to_flac(
        self, image: str, sample_format: str
    ) -> None:
        """WAV, MP3, Ogg Vorbis, FLAC and AAC uploads decode to FLAC too.

        These formats usually pass straight through, but a model that declines
        one, or an upload the sniffer maps to a format the model rejects, sends
        it down the same fallback; the Ogg Vorbis case doubles as the decode
        side of speech synthesis above Polly's 16 kHz pcm cap.

        Ref: stdapi/models/audio/_default.py:AudioModel._audio_content_block
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        sample = _input_sample(sample_format)

        output = _image_ffmpeg(
            image, _ffmpeg_args("flac", None, None, None, None), sample
        )

        _assert_signature("flac", output)


class TestFinalImageRuntime:
    """The final image still provides everything the build stripped out of it.

    The builder stage smoke-tests its own filesystem, but the image that ships
    is assembled from copies: a shared library, a libmagic database, a timezone
    database or a package that needs its sources back can be missing there and
    nowhere else. Each case below runs inside the shipped image, with no network
    and no credentials, and compares what it finds against the application's own
    code.

    Ref: https://docs.podman.io/en/latest/markdown/podman-image-inspect.1.html
         stdapi/main.py:app
         stdapi/aws.py:AWSConnectionManager
    """

    def test_the_served_application_imports(self, image: str) -> None:
        """``stdapi.main`` imports in the final image and builds its routes.

        The widest check of the lot: it fails on a missing shared library, a
        data file the prune took with it, a module whose bytecode never
        compiled, and a package that reads its own deleted source at import.

        Ref: https://docs.python.org/3/library/importlib.html
             stdapi/main.py:app
        """
        routes = _image_python(image, _IMPORT_PROGRAM)

        assert int(routes.strip()) > 0, "the application exposed no route"

    def test_every_aws_client_the_application_builds_constructs(
        self, image: str
    ) -> None:
        """Every service the application names still has its ``botocore/data``.

        The image keeps the service data of an allowlist and deletes the rest,
        which is only safe while the allowlist covers what the server uses. The
        names come from the server's own source, so a service it starts calling
        fails here — with ``UnknownServiceError`` — until the image ships its
        data, which is the failure the pruning would otherwise cause in
        production.

        Ref: https://botocore.amazonaws.com/v1/documentation/api/latest/reference/session.html
             stdapi/aws.py:get_client
        """
        services = _application_aws_services()
        # The derivation is static: a rename that made it silently find nothing
        # would turn this test green while covering none of the pruning.
        assert "bedrock-runtime" in services, (
            f"no Bedrock client was derived from the application source: {services}"
        )

        constructed = _image_python(
            image, _CLIENT_PROGRAM, _CLIENT_PROBE_REGION, *services
        )

        assert int(constructed.strip()) == len(services)

    def test_libmagic_identifies_the_uploads_the_application_maps(
        self, image: str
    ) -> None:
        """Libmagic and its database answer with the MIME types the routes key on.

        The upload path does not ask whether a type was detected, it looks the
        detected subtype up in the Bedrock format tables; a libmagic without its
        compiled database answers ``application/octet-stream`` for everything
        and every upload becomes unroutable. The application's own detection
        function is called, so the temporary-file fallback it uses is covered
        too, and the answers are checked against those very tables.

        Ref: https://man7.org/linux/man-pages/man3/libmagic.3.html
             stdapi/input_file.py:_magic_detect
             stdapi/aws_bedrock.py:MIME_TYPES_TO_AUDIO_TYPE
        """
        samples = (_wav_sample(), _png_sample(), _TEXT_SAMPLE)
        framed = b"".join(struct.pack(">L", len(s)) + s for s in samples)

        detected = json.loads(_image_python(image, _MAGIC_PROGRAM, stdin=framed))

        audio, picture, document = detected
        assert audio == _WAV_MIME_TYPE, f"a WAV upload was sniffed as '{audio}'"
        assert MIME_TYPES_TO_AUDIO_TYPE[audio.partition("/")[2]] == "wav"
        assert picture == _PNG_MIME_TYPE, f"a PNG upload was sniffed as '{picture}'"
        assert document == _TEXT_MIME_TYPE, f"text was sniffed as '{document}'"
        assert MIME_TYPES_TO_DOCUMENT_TYPE[document.partition("/")[2]] == "txt"

    def test_the_application_ships_only_compiled_modules(self, image: str) -> None:
        """The application directory holds ``.pyc`` files and no source at all.

        ``compileall -b`` writes each module's bytecode beside it instead of
        into a ``__pycache__`` directory, and the sources are then deleted; a
        stale ``.py``, a leftover cache directory or a missing ``.pyc`` all mean
        the layout the image is meant to ship was not produced.

        Ref: https://docs.python.org/3/library/compileall.html
             stdapi/main.py:app
        """
        layout = json.loads(_image_python(image, _LAYOUT_PROGRAM))

        assert layout["sources"] == [], (
            f"source files remain under {layout['root']}: {layout['sources'][:10]}"
        )
        assert layout["caches"] == [], (
            f"'__pycache__' directories remain under {layout['root']}: "
            f"{layout['caches'][:10]}"
        )
        assert layout["compiled"] > 0, f"no bytecode was found under {layout['root']}"

    def test_the_documentation_icon_ships_in_the_image(self, image: str) -> None:
        """The icon the documentation pages are branded with is package data.

        The build copies the application tree and then deletes every ``.py``
        file in it, so anything else the package carries has to survive that
        rewrite: an icon missing here is a ``/docs`` page with a broken image
        and no way to fetch the original, which is the whole point of serving
        it from the gateway.

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/184
             stdapi/routes/core_docs.py:_FAVICON_RESPONSE
        """
        expected = (files("stdapi") / "favicon.svg").read_bytes()

        assert int(_image_python(image, _FAVICON_PROGRAM)) == len(expected)

    def test_the_documentation_assets_ship_in_the_image(self, image: str) -> None:
        """Every pinned Swagger UI and ReDoc file is present, and is its own digest.

        This is the only check that the build's fetch step ran at all: a source
        checkout has none of these files, and the pages silently fall back to
        the publisher there. Reading them also proves the runtime user can, and
        comparing the digests proves the bytes are the reviewed ones rather than
        whatever a CDN happened to answer with during the build.

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
             stdapi/docs_assets/__init__.py
             Dockerfile
        """
        fetched = json.loads(_image_python(image, _DOCS_ASSET_PROGRAM))

        assert fetched == {name: asset.sha256 for name, asset in BROWSER_ASSETS.items()}

    def test_the_documentation_pages_load_nothing_from_a_third_party(
        self, image: str
    ) -> None:
        """Both pages render, in a container with no network, naming no outside host.

        The pages are rendered through the served application itself, with the
        container cut off from every network: whatever they reference, a browser
        in an air-gapped deployment has to be able to fetch from the gateway.
        Anything still pointing at a CDN renders as a blank page there.

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
             stdapi/routes/core_docs.py
        """
        pages = json.loads(
            _image_python(image, _DOCS_PAGE_PROGRAM, *_DOCUMENTATION_PAGES)
        )

        for path, names in _DOCUMENTATION_PAGES.items():
            page = pages[path]
            assert page["status"] == 200, f"'{path}' answered {page['status']}"
            for name in names:
                assert f'"{ASSETS_PATH}/{name}"' in page["body"], (
                    f"'{path}' does not load '{name}' from the gateway"
                )
            for host in _THIRD_PARTY_HOSTS:
                assert host not in page["body"], (
                    f"'{path}' still reaches '{host}' in the built image"
                )

    def test_the_distribution_metadata_still_resolves(self, image: str) -> None:
        """Every distribution the runtime queries still reports a version.

        The build empties each ``.dist-info`` down to its ``METADATA`` file, so
        a package asking for its own version at runtime — ``fastapi_mcp`` does,
        for ``mcp`` — depends on that file surviving intact.

        Ref: https://docs.python.org/3/library/importlib.metadata.html
             Dockerfile
        """
        versions = json.loads(
            _image_python(image, _METADATA_PROGRAM, *_METADATA_DISTRIBUTIONS)
        )

        assert sorted(versions) == sorted(_METADATA_DISTRIBUTIONS)
        for name, version in versions.items():
            assert version, f"'{name}' reports no version in the image"

    def test_the_timezone_database_is_usable(self, image: str) -> None:
        """A regional timezone constructs, so the configured one can too.

        The server validates its ``timezone`` setting against the zones the
        database offers and stamps every request date with the result, so an
        image without ``tzdata`` rejects every configuration but the default.

        Ref: https://docs.python.org/3/library/zoneinfo.html
             stdapi/config.py:_Settings._parse_timezone
        """
        report = json.loads(_image_python(image, _TIMEZONE_PROGRAM, _TIMEZONE_KEY))

        assert report["key"] == _TIMEZONE_KEY
        assert report["available"] > 0, "the image offers no timezone at all"

    def test_the_image_does_not_run_as_root(self, image: str) -> None:
        """The image runs its command as an unprivileged user.

        The two images use different accounts, so only the property that matters
        is asserted: whatever the server runs as, it is not root.

        Ref: https://docs.docker.com/reference/dockerfile/#user
             Dockerfile
        """
        euid = int(_image_python(image, _EUID_PROGRAM).strip())

        assert euid != 0, "the image runs its command as root"

    def test_the_declared_healthcheck_is_executable(self, image: str) -> None:
        """An image declaring a healthcheck ships the binaries it invokes.

        The probe is a shell command naming a tool the base image happens to
        provide, which nothing else in the image needs; a base change that drops
        it makes every container report unhealthy forever, and only here. Images
        declaring no healthcheck — their orchestrator probes ``/health`` itself
        — are skipped.

        Ref: https://docs.docker.com/reference/dockerfile/#healthcheck
             stdapi/routes/core_root.py:health_check
        """
        test = _image_healthcheck(image)
        if not test:
            pytest.skip("the image declares no healthcheck")
        binaries = _healthcheck_binaries(test)
        assert binaries, f"no executable could be read from the healthcheck {test}"

        resolved = json.loads(_image_python(image, _WHICH_PROGRAM, *binaries))

        missing = sorted(name for name, path in resolved.items() if not path)
        assert not missing, (
            f"the healthcheck {test} needs {missing}, which the image does not ship"
        )

    @pytest.mark.parametrize(
        "trusted_hosts", _TRUSTED_HOSTS_CASES.values(), ids=_TRUSTED_HOSTS_CASES
    )
    def test_the_healthcheck_passes_under_host_validation(
        self, image: str, trusted_hosts: tuple[str, ...]
    ) -> None:
        """The declared probe stays healthy under every ``TRUSTED_HOSTS`` setting.

        ``TRUSTED_HOSTS`` makes the server answer 400 to every Host header it
        does not name, ahead of routing and ``/health`` included, so a probe
        announcing ``localhost`` fails for a deployment naming only its public
        names — and the orchestrator kills a container that is serving traffic
        correctly. The header the probe sends is judged by the middleware the
        server itself installs, so no rule of it is restated here.

        Ref: https://www.starlette.io/middleware/#trustedhostmiddleware
             stdapi/main.py:app
             stdapi/config.py:_Settings.trusted_hosts
        """
        if not _image_healthcheck(image):
            pytest.skip("the image declares no healthcheck")

        probe = _healthcheck_probe(image, trusted_hosts)

        assert probe["exit"] == 0, (
            f"the probe failed with TRUSTED_HOSTS={list(trusted_hosts)}:"
            f"\n{probe['output']}"
        )
        assert probe.get("path") == "/health", (
            f"the probe requested {probe.get('path')!r}:\n{probe['output']}"
        )
        host = str(probe.get("host", ""))
        assert host, f"the probe announced no Host header:\n{probe['output']}"
        assert not trusted_hosts or _host_is_trusted(host, trusted_hosts), (
            f"the probe announced 'Host: {host}', which the server rejects with "
            f"400 when TRUSTED_HOSTS={list(trusted_hosts)}"
        )

    def test_every_licence_file_the_packages_declare_ships(self, image: str) -> None:
        """Each distribution keeps the licence files its own METADATA declares.

        The image strips every ``.dist-info`` down to what it needs, and
        publishing it is a binary redistribution: the Apache-2.0, BSD and MIT
        terms of those packages all require their licence text and notices to
        travel along. The expectation is each package's own ``License-File``
        declarations, so a dependency added with a licence file is covered from
        the moment it is installed.

        Ref: https://packaging.python.org/en/latest/specifications/core-metadata/#license-file-multiple-use
             Dockerfile
        """
        report = json.loads(_image_python(image, _LICENSE_PROGRAM))
        declared = report["declared"]
        # The derivation is metadata-driven: a stripped METADATA, or a parse
        # that stopped matching, would declare nothing and check nothing.
        for name in _LICENSED_DISTRIBUTIONS:
            assert declared.get(name), (
                f"'{name}' declares no licence file in the image, so this test "
                f"proves nothing about the {len(declared)} distributions installed"
            )

        assert report["missing"] == [], (
            "the image drops licence files its own packages declare: "
            f"{report['missing']}"
        )

    def test_the_image_ships_the_licences_of_what_it_redistributes(
        self, image: str
    ) -> None:
        """The application's own licence and ffmpeg's travel with the binaries.

        The image is published as a whole: the served application under its
        edition's licence, and the ffmpeg it builds from source under the LGPL.
        Both texts must be in the image, and readable by the unprivileged user
        it runs as.

        Ref: https://www.gnu.org/licenses/agpl-3.0.html
             https://www.ffmpeg.org/legal.html
             Dockerfile
        """
        listing = json.loads(
            _image_python(image, _LICENSE_DIRECTORY_PROGRAM, *_LICENSE_DIRECTORIES)
        )

        for directory in _LICENSE_DIRECTORIES:
            files = listing[directory]
            assert files, f"the image ships no licence text in '{directory}'"
            empty = sorted(name for name, size in files.items() if not size)
            assert not empty, f"'{directory}' holds unreadable or empty files: {empty}"


class TestServerBoot:
    """The image starts a server that serves the routes the suite exercises.

    Building on the transcode checks, this runs the ``image``-marked subset of
    the real test suite against the started container: the routes whose failure
    mode is an image-build regression rather than a code defect.

    Ref: Dockerfile
         stdapi/main.py:app
    """

    def test_the_image_serves_the_image_marked_tests(self, image: str) -> None:
        """The server boots, answers its probes and passes the marked subset.

        AWS credentials reach the container through a private env file, never a
        command line and never a failure message. The ambient ``AWS_*`` settings
        ride along, since the marked tests exercise the gateway this session is
        configured for.

        Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html
             stdapi/routes/core_root.py:health_check
        """
        credentials = _aws_credentials()
        if credentials is None:
            pytest.skip("no AWS credentials resolvable; the server exits without them")
        forwarded = {**_forwarded_settings(), **credentials}
        container = f"stdapi-image-test-{token_hex(4)}"

        with _env_file(forwarded) as env_file:
            started = _engine_run(
                [
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--publish",
                    f"127.0.0.1::{_SERVER_PORT}",
                    "--env-file",
                    env_file,
                    image,
                ]
            )
        assert started.returncode == 0, (
            f"the container did not start:\n{_tail(started.stderr)}"
        )
        try:
            port = _published_port(container)
            _wait_for_health(container, port)
            assert _probe(port, "/") == 200, (
                f"the root route did not answer:\n{_container_logs(container)}"
            )

            result = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-m",
                    "image",
                    "--server-url",
                    f"http://127.0.0.1:{port}",
                    "-q",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                timeout=_SUITE_TIMEOUT,
                env={**environ, **credentials},
            )

            report = _tail(result.stdout + result.stderr)
            assert result.returncode == 0, (
                f"the 'image' tests failed against the container:\n{report}\n"
                f"{_container_logs(container)}"
            )
            # A run that skipped everything also exits zero, and would report a
            # container nothing was ever asked of as healthy.
            assert " passed" in report, (
                f"no 'image' test ran against the container:\n{report}"
            )
        finally:
            # Removal is explicit rather than "--rm": a container that dies at
            # startup must still have logs to attach to the failure above.
            _engine_run(["stop", "-t", _STOP_TIMEOUT, container])
            _engine_run(["rm", "-f", container])
