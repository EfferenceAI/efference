#!/usr/bin/env python3
################################################################################
#
# File:      mcap_to_video.py
# Purpose:   Extract the video from an MCAP recording as an MP4 or a raw
#            H.264/H.265 elementary stream.
# Author:    Calvin Nguyen
#
# Copyright (c) 2026, Remnant Robotics, Inc. All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
################################################################################

import argparse
import contextlib
import math
import shutil
import statistics
import subprocess
import sys
import tempfile

try:
    from mcap.reader import NonSeekingReader
except ImportError:
    raise SystemExit("error: this tool needs the 'mcap' package: pip install mcap")

MCAP_MAGIC = b"\x89MCAP0\r\n"
ENCRYPTED_MAGIC = b"EFE1"

VIDEO_SCHEMA = "foxglove.CompressedVideo"
RAW_SCHEMA = "foxglove.RawImage"

DEMUXER = {"h265": "hevc", "h264": "h264", "jpeg": "mjpeg"}

# Apple players need hvc1; ffmpeg writes hev1.
CODEC_TAG = {"h265": "hvc1"}

PARAM_SET_NALS = {"h265": (32, 33, 34), "h264": (7, 8)}

MP4_SUFFIXES = (".mp4", ".m4v", ".mov")

DEFAULT_FPS = 30.0


def die(msg, hint=None):
    raise SystemExit(f"error: {msg}" + (f"\n       {hint}" if hint else ""))


def video_fields(buf):
    """Return (data, format) from a foxglove.CompressedVideo message: 3=data, 4=format."""
    data = b""
    fmt = ""
    i, n = 0, len(buf)
    while i < n:
        key = buf[i]
        i += 1
        fno, wt = key >> 3, key & 7
        if wt == 0:  # varint
            while i < n and buf[i] & 0x80:
                i += 1
            i += 1
        elif wt == 2:  # length-delimited
            ln = 0
            sh = 0
            while True:
                if i >= n:
                    return data, fmt
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << sh
                sh += 7
                if not (b & 0x80):
                    break
            if i + ln > n:
                return data, fmt
            if fno == 3:
                data = bytes(buf[i:i + ln])
            elif fno == 4:
                fmt = bytes(buf[i:i + ln]).decode("ascii", "replace")
            i += ln
        elif wt == 5:  # fixed32
            i += 4
        elif wt == 1:  # fixed64
            i += 8
        else:
            break
    return data, fmt


def has_parameter_set(frame, codec):
    """True if a decoder can start on this frame."""
    wanted = PARAM_SET_NALS.get(codec)
    if wanted is None:
        return True  # JPEG frames are each self-contained
    i, n = 0, len(frame)
    while True:
        i = frame.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= n:
            return False
        head = frame[i + 3]
        nal = (head >> 1) & 0x3F if codec == "h265" else head & 0x1F
        if nal in wanted:
            return True
        i += 3


class Recording:
    """One .mcap file. `truncated` is set if it ends short."""

    def __init__(self, path):
        self.path = path
        self.truncated = None

    def check_readable(self):
        """Exit on the inputs that open but hold no readable recording."""
        try:
            with open(self.path, "rb") as f:
                head = f.read(len(MCAP_MAGIC))
        except OSError as e:
            die(f"cannot read '{self.path}': {e.strerror}")
        if head.startswith(ENCRYPTED_MAGIC):
            die(f"'{self.path}' is an encrypted recording",
                f"Decrypt it first:  ef-decrypt {self.path} <key-file> plain.mcap")
        if not head.startswith(MCAP_MAGIC):
            die(f"'{self.path}' is not an MCAP recording")

    def records(self):
        """Yield (schema, message). Streaming: recordings are unchunked, and may be short."""
        with open(self.path, "rb") as f:
            records = NonSeekingReader(f).iter_messages(log_time_order=False)
            while True:
                try:
                    schema, _channel, message = next(records)
                except StopIteration:
                    return
                except Exception as e:
                    self.truncated = str(e) or type(e).__name__
                    return
                yield schema, message

    def frames(self):
        """Yield (data, log_time) per video frame; data is empty if damaged."""
        for schema, message in self.records():
            if schema and schema.name == VIDEO_SCHEMA:
                data, _fmt = video_fields(message.data)
                yield data, message.log_time


def probe(rec, measure_fps):
    """Return (codec, fps). fps is None unless measured, which reads the whole file."""
    codec = ""
    saw_raw = saw_video = False
    gaps = []
    prev = None
    for schema, message in rec.records():
        if not schema:
            continue
        if schema.name == VIDEO_SCHEMA:
            saw_video = True
            if not codec:
                _data, codec = video_fields(message.data)
            if not measure_fps:
                if codec:
                    return codec, None
                continue
            if prev is not None and message.log_time > prev:
                gaps.append(message.log_time - prev)
            prev = message.log_time
        elif schema.name == RAW_SCHEMA:
            saw_raw = True

    if codec:
        return codec, (1e9 / statistics.median(gaps) if gaps else None)
    if saw_video:
        die(f"every video frame in '{rec.path}' is damaged")
    if saw_raw:
        die(f"'{rec.path}' holds raw NV12 frames, which are not a coded video stream",
            "Record h264 or h265 to convert to video.")
    if rec.truncated:
        die(f"'{rec.path}' is incomplete and holds no whole video frame",
            "Download it again.")
    die(f"'{rec.path}' holds no video frames")


def resolve_format(requested, output):
    if requested != "auto":
        return requested
    return "mp4" if output.lower().endswith(MP4_SUFFIXES) else "raw"


@contextlib.contextmanager
def open_sink(out_format, output, codec, fps):
    """Yield a write(bytes) for the chosen output format."""
    if out_format == "raw":
        try:
            handle = open(output, "wb")
        except OSError as e:
            die(f"cannot write '{output}': {e.strerror}")
        with handle:
            yield handle.write
        return

    demuxer = DEMUXER.get(codec)
    if demuxer is None:
        die(f"cannot mux '{codec or 'unknown'}' into MP4",
            "Write the elementary stream instead.")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", demuxer, "-r", f"{fps:g}", "-i", "-", "-c", "copy"]
    if codec in CODEC_TAG:
        cmd += ["-tag:v", CODEC_TAG[codec]]
    # ffmpeg has no "--" terminator, so a leading dash would read as an option.
    cmd.append(f"./{output}" if output.startswith("-") else output)

    # A file, never a pipe: nothing drains it until the last frame is written.
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    try:
        yield proc.stdin.write
    except BrokenPipeError:
        pass  # ffmpeg died early; its log says why
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        rc = proc.wait()
        log.seek(0)
        err = log.read().decode("utf-8", "replace").strip()
        log.close()
        if rc != 0:
            die(f"writing '{output}' failed: "
                f"{err.splitlines()[-1] if err else f'ffmpeg exited {rc}'}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Extract the video from an MCAP recording.",
        epilog="The output extension picks the format: .mp4 is muxed, anything "
               "else is written as the raw elementary stream.")
    p.add_argument("input", help="recording to read (.mcap)")
    p.add_argument("output", help="file to write (.mp4, .h265, .h264)")
    p.add_argument("--format", choices=("auto", "mp4", "raw"), default="auto",
                   help="override the format implied by the output extension")
    p.add_argument("--fps", default=f"{DEFAULT_FPS:g}",
                   help="frame rate for the MP4: a number, or 'auto' to measure "
                        "it from the recording (default: %(default)s)")
    p.add_argument("--allow-mid-gop", action="store_true",
                   help="keep leading frames that arrive before the first "
                        "parameter set instead of skipping them")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    fps = None
    if args.fps != "auto":
        try:
            fps = float(args.fps)
        except ValueError:
            die(f"--fps: '{args.fps}' is not a number or 'auto'")
        if not math.isfinite(fps) or fps <= 0:
            die("--fps: must be a finite number greater than zero")

    rec = Recording(args.input)
    rec.check_readable()

    out_format = resolve_format(args.format, args.output)
    if out_format == "mp4" and not shutil.which("ffmpeg"):
        die("writing MP4 needs ffmpeg on PATH",
            "Install it, or write the elementary stream with --format raw.")

    codec, measured = probe(rec, fps is None)
    if fps is None:
        fps = measured or DEFAULT_FPS
        if measured is None:
            print(f"note: too few frames to measure a rate, using {fps:g} fps",
                  file=sys.stderr)

    frames = skipped = damaged = 0
    started = args.allow_mid_gop
    with open_sink(out_format, args.output, codec, fps) as write:
        for data, _log_time in rec.frames():
            if not data:
                damaged += 1
                continue
            if not started:
                if not has_parameter_set(data, codec):
                    skipped += 1
                    continue
                started = True
            write(data)
            frames += 1

    if not frames:
        die(f"no decodable frames in '{args.input}'")
    if rec.truncated:
        print(f"warning: '{args.input}' is incomplete; converted everything "
              f"that was readable", file=sys.stderr)

    name = codec or "video"
    detail = f"{name}, {fps:g} fps" if out_format == "mp4" else name
    print(f"wrote {frames} frames ({detail}) -> {args.output}")
    if skipped:
        print(f"  skipped {skipped} leading frames before the first parameter set")
    if damaged:
        print(f"  skipped {damaged} damaged frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
