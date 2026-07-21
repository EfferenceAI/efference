#!/usr/bin/env python3
################################################################################
#
# File:      mcap_to_video.py
# Purpose:   Extract the raw H.264/H.265 elementary stream from an MCAP recording.
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

import sys
from mcap.reader import make_reader

def field3_bytes(buf):
    # foxglove.CompressedVideo: 1=timestamp(msg) 2=frame_id(str) 3=data(bytes) 4=format(str)
    # walk the protobuf wire, return the bytes of field 3.
    i, n = 0, len(buf)
    while i < n:
        key = buf[i]; i += 1
        fno, wt = key >> 3, key & 7
        if wt == 0:      # varint
            while buf[i] & 0x80: i += 1
            i += 1
        elif wt == 2:    # length-delimited
            ln = 0; sh = 0
            while True:
                b = buf[i]; i += 1
                ln |= (b & 0x7f) << sh; sh += 7
                if not (b & 0x80): break
            if fno == 3:
                return bytes(buf[i:i+ln])
            i += ln
        elif wt == 5: i += 4
        elif wt == 1: i += 8
        else: break
    return b""

def main():
    if len(sys.argv) != 3:
        print("usage: mcap_to_video.py <input.mcap> <output.h264|.h265>",
              file=sys.stderr)
        return 2
    inp, outp = sys.argv[1], sys.argv[2]
    frames = 0
    try:
        with open(inp, "rb") as f, open(outp, "wb") as o:
            for schema, channel, message in make_reader(f).iter_messages():
                if schema and schema.name == "foxglove.CompressedVideo":
                    o.write(field3_bytes(message.data)); frames += 1
    except FileNotFoundError as e:
        print(f"error: {e.filename}: no such file", file=sys.stderr)
        return 1
    except (OSError, ValueError, IndexError) as e:   # IndexError: truncated/malformed protobuf
        print(f"error: could not read '{inp}': {e}", file=sys.stderr)
        return 1
    print(f"wrote {frames} frames -> {outp}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
