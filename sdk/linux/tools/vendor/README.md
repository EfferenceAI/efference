# Vendored recording-format sources

Byte-for-byte copies of the device's encrypted-recording implementation, carried
here so `ef-decrypt` can read a recording without a firmware checkout. Do not
edit them: the device writes what this code reads, and an edit on one side only
changes what the two agree on.

| File | Upstream path (firmware repo) |
|---|---|
| `enc_sink.c` / `enc_sink.h` | `project/app/efference-capture/rec/` |
| `efr_keyid.c` / `efr_keyid.h` | `project/app/libefr/{src,include}/` |

Copied from firmware `09d1788bf` (v00.09.15).

```
524a6033ebf3fbfee527ecb82685d8b714f5c364d14e943aafafc69ba16324f1  efr_keyid.c
309cb1e151f64d24e9ed9ba7599a76d0581e50f93e6a34ef050d334e23f1ca26  efr_keyid.h
d9fcaa61b11f0e44efaebc257459c7d450b616bc359e1107b9399d1667113fa3  enc_sink.c
437371b3ef1874ee457e5c0f0121003d0dd3ae01050110c95ae1eefc963d7c94  enc_sink.h
```

Nothing checks these hashes automatically. Refresh the copies, and this table,
whenever the format changes upstream. A format that has genuinely moved on
announces itself rather than misreading: the header carries a version at 0x04 and
an algorithm id at 0x0C, and `enc_decrypt_fd()` returns `ENOTSUP` for an id it
does not implement. The format itself is specified in the header comment of
`enc_sink.h`.
