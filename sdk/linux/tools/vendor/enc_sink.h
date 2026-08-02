/* Encrypting byte sink — AES-256-GCM over an arbitrary output stream.
 *
 * Container-agnostic on purpose: it sees bytes and an fd, never MCAP or MP4, so
 * the crypto layer stays independent of whichever container a recording uses.
 *
 * Chunked rather than whole-file for two reasons: a streaming writer cannot
 * buffer a whole recording, and a file truncated by power loss must still
 * decrypt up to its last complete chunk.
 *
 * FORMAT (little-endian, all offsets bytes)
 *
 *   header, 64 B:
 *     0x00  4   magic 'E','F','E','1'
 *     0x04  2   version = 1
 *     0x06  2   header_len = 64
 *     0x08  4   chunk_bytes   (max plaintext per chunk)
 *     0x0C  4   flags: byte 0 = algorithm id, bytes 1-3 reserved (0)
 *     0x10 12   base_nonce (random per file)
 *     0x1C  4   key_id = first 4 bytes of SHA-256(key), 0 if unknown
 *     0x20 32   device_id, NUL-padded ASCII
 *
 * Both the algorithm id and key_id were reserved bytes through version 1, so
 * files written before them carry 0 in each: algorithm 0 reads as AES-256-GCM
 * and key_id 0 as "unnamed". No version bump, and old files still decrypt.
 *
 *   then one record per chunk:
 *     0x00  4   plain_len
 *     0x04  1   cflags: bit0 = final chunk
 *     0x05  3   reserved (0)      -- keeps ciphertext 8-B aligned
 *     0x08  N   ciphertext (N == plain_len)
 *     8+N  16   GCM tag
 *
 * Per-chunk nonce = base_nonce with its trailing 8 bytes XORed by the big-endian
 * chunk index, so a repeat needs a 96-bit base_nonce collision.
 *
 * AAD = header || be64(index) || the 8-byte record prefix. Binding the header
 * makes any edit to chunk_bytes/device_id/base_nonce fail every tag; binding the
 * index and length blocks reordering, splicing, and truncating a chunk short.
 *
 * The final chunk carries cflags bit0 (and may be zero-length), so a decryptor
 * distinguishes a clean close from a truncated file. An attacker cannot forge
 * the end marker, because cflags is covered by the tag.
 */

#ifndef EFR_ENC_SINK_H
#define EFR_ENC_SINK_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "efr_keyid.h"   /* key_id: one derivation, shared with the control endpoint */

#ifdef __cplusplus
extern "C" {
#endif

#define ENC_KEY_BYTES      32
#define ENC_NONCE_BYTES    12
#define ENC_TAG_BYTES      16
#define ENC_HEADER_BYTES   64
#define ENC_PREFIX_BYTES    8
#define ENC_DEVICE_ID_MAX  32

/* Algorithm ids, matching ef.proto's EncryptionAlgorithm so the wire value and
 * the on-disk byte are the same number. 0 means "written before the field
 * existed", which was always AES-256-GCM. */
#define ENC_ALG_UNSPECIFIED  0
#define ENC_ALG_AES_256_GCM  1

#define ENC_KEY_ID_BYTES    EFR_KEY_ID_BYTES
#define ENC_KEY_ID_STRLEN   EFR_KEY_ID_STRLEN

#define ENC_CHUNK_DEFAULT  (1u << 20)   /* 1 MiB */
#define ENC_CHUNK_MIN      4096u
#define ENC_CHUNK_MAX      (16u << 20)

struct enc_sink;
typedef struct enc_sink enc_sink_t;

/* Load a 32-byte key from `path`. Accepts raw 32 bytes or 64 hex chars with an
 * optional trailing newline. Returns 0, or -1 with errno set (EINVAL on a
 * malformed file). */
int enc_key_load(const char *path, uint8_t key[ENC_KEY_BYTES]);

/* key_id is the one header field this TU does not define. It comes from libefr's
 * efr_key_id(), because the control endpoint has to stamp and report the same
 * value; call that directly rather than wrapping it here, so there is no second
 * implementation to drift. */

/* Wrap `fd`, writing the header immediately. The caller keeps ownership of fd
 * and closes it after enc_sink_finish/abort. `device_id` may be NULL. Pass 0 for
 * chunk_bytes to get ENC_CHUNK_DEFAULT; other values are clamped to
 * [ENC_CHUNK_MIN, ENC_CHUNK_MAX]. Returns NULL on failure. */
enc_sink_t *enc_sink_open(int fd, const uint8_t key[ENC_KEY_BYTES],
                          const char *device_id, uint32_t chunk_bytes);

/* Buffer and seal. Returns len on success, -1 on error (sticky: see
 * enc_sink_err). Short writes are not possible; a partial write(2) is retried. */
ssize_t enc_sink_write(enc_sink_t *s, const void *buf, size_t len);

/* Seal whatever is pending as a non-final chunk, so an fsync of the fd actually
 * commits it. No-op when nothing is buffered. Callers that fdatasync on a cadence
 * must call this first, or the sync covers stale bytes and a power cut loses up
 * to chunk_bytes. Sealing early only costs 24 B of framing per extra chunk. */
int enc_sink_flush(enc_sink_t *s);

/* Seal the pending bytes as the final chunk (emitting a zero-length final chunk
 * if nothing is pending), then free *s and NULL it. Returns 0 on success, -1 if
 * any write failed. Does not close the fd.
 *
 * `out_cipher_bytes` (may be NULL) receives the final on-disk size; it is the
 * only way to learn it, since the handle is gone on return and the caller
 * typically needs it to ftruncate. */
int enc_sink_finish(enc_sink_t **s, uint64_t *out_cipher_bytes);

/* Free without sealing; the file stays a torso with no end marker. */
void enc_sink_abort(enc_sink_t **s);

/* Sticky error flag: nonzero once any write or seal failed. */
int enc_sink_err(const enc_sink_t *s);

uint64_t enc_sink_plain_bytes (const enc_sink_t *s);
uint64_t enc_sink_cipher_bytes(const enc_sink_t *s);

/* ---- decrypt side (host tool + tests; same TU so one place owns the format) */

struct enc_dec_info {
    char     device_id[ENC_DEVICE_ID_MAX + 1];
    uint32_t chunk_bytes;
    uint64_t chunks;        /* chunks successfully verified */
    uint64_t plain_bytes;
    int      clean_eof;     /* 1 if the final-chunk marker was seen */
    int      tag_failed;    /* 1 if a chunk failed authentication; clean_eof==0 without
                             * this means the ciphertext simply ends early (truncation) */
    uint8_t  algorithm;     /* id from the header; 0 == AES-256-GCM (pre-field file) */
    char     key_id[ENC_KEY_ID_STRLEN];   /* "" when the file predates the field */
};

/* Decrypt in_fd to out_fd. Stops at the first chunk that fails its tag or is
 * incomplete, having already written every chunk before it, and reports
 * clean_eof = 0. `info` may be NULL.
 *
 * Returns 0 when the stream ended cleanly, 1 when it was truncated or a tag
 * failed (partial output is still valid), -1 on I/O error or a bad header.
 *
 * Two header checks fail EARLY rather than letting every chunk fail its tag,
 * because "wrong key" and "algorithm this build cannot do" are worth saying out
 * loud: an unsupported algorithm gives ENOTSUP, and a header key_id that names a
 * different key than the one supplied gives EKEYREJECTED. `info` is filled in
 * both cases, so a caller can report which key the file actually wants. */
int enc_decrypt_fd(int in_fd, int out_fd, const uint8_t key[ENC_KEY_BYTES],
                   struct enc_dec_info *info);

#ifdef __cplusplus
}
#endif

#endif /* EFR_ENC_SINK_H */
