# Vendored recording-format sources

Byte-for-byte copies of the device's encrypted-recording implementation, carried
here so `ef-decrypt` can read a recording without a firmware checkout. Do not
edit them: the device writes what this code reads, and an edit on one side only
changes what the two agree on.

| File | Provides |
|---|---|
| `enc_sink.c` / `enc_sink.h` | the encrypted-container reader |
| `efr_keyid.c` / `efr_keyid.h` | key-ID derivation |

`enc_dec_info.tag_failed` distinguishes a chunk-authentication failure from a
truncated file.

```
524a6033ebf3fbfee527ecb82685d8b714f5c364d14e943aafafc69ba16324f1  efr_keyid.c
309cb1e151f64d24e9ed9ba7599a76d0581e50f93e6a34ef050d334e23f1ca26  efr_keyid.h
52f8e1f7fe062843f24eb59cc9ac4c8ad62f1224b695ff8b28cc8e4f5f10f10a  enc_sink.c
c72fd1907cbb3bdc0a90bfd6d44a45ace6803298da9b6766babf17b03ac8895f  enc_sink.h
```

Refresh the copies, and this table, whenever the format changes upstream. A
format that has genuinely moved on
announces itself rather than misreading: the header carries a version at 0x04 and
an algorithm id at 0x0C, and `enc_decrypt_fd()` returns `ENOTSUP` for an id it
does not implement. The format itself is specified in the header comment of
`enc_sink.h`.
