/* AES-256-GCM chunked byte sink. Format spec lives in enc_sink.h. */

#include "enc_sink.h"
#include "efr_keyid.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <openssl/evp.h>
#include <openssl/rand.h>

#define ENC_MAGIC0 'E'
#define ENC_MAGIC1 'F'
#define ENC_MAGIC2 'E'
#define ENC_MAGIC3 '1'
#define ENC_VERSION 1

#define CFLAG_FINAL 0x01

struct enc_sink {
    int       fd;
    uint8_t   key[ENC_KEY_BYTES];
    uint8_t   hdr[ENC_HEADER_BYTES];   /* kept for AAD */
    uint8_t   base_nonce[ENC_NONCE_BYTES];
    uint32_t  chunk_bytes;
    uint8_t  *buf;                     /* pending plaintext, chunk_bytes cap */
    uint8_t  *ct;                      /* ciphertext scratch, same cap */
    uint32_t  buf_len;
    uint64_t  index;                   /* next chunk index */
    uint64_t  plain_bytes;
    uint64_t  cipher_bytes;
    int       err;
    EVP_CIPHER_CTX *ctx;               /* reused across chunks */
};

/* ---- small helpers ---- */

static void put_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}

static uint16_t get_u16le(const uint8_t *p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

static void put_u64be(uint8_t *p, uint64_t v) {
    for (int i = 7; i >= 0; i--) { p[i] = (uint8_t)v; v >>= 8; }
}

static int write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len) {
        ssize_t n = write(fd, p, len);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) { errno = EIO; return -1; }
        p += n; len -= (size_t)n;
    }
    return 0;
}

/* Returns bytes read: len on success, 0 at clean EOF, >0 && <len on a short
 * tail, -1 on error. */
static ssize_t read_upto(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, p + got, len - got);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) break;
        got += (size_t)n;
    }
    return (ssize_t)got;
}

/* nonce_i = base_nonce, trailing 8 bytes XOR be64(index) */
static void chunk_nonce(const uint8_t base[ENC_NONCE_BYTES], uint64_t index,
                        uint8_t out[ENC_NONCE_BYTES]) {
    uint8_t ctr[8];
    put_u64be(ctr, index);
    memcpy(out, base, ENC_NONCE_BYTES);
    for (int i = 0; i < 8; i++) out[4 + i] ^= ctr[i];
}

/* AAD = header || be64(index) || record prefix */
static void build_aad(const uint8_t hdr[ENC_HEADER_BYTES], uint64_t index,
                      const uint8_t prefix[ENC_PREFIX_BYTES],
                      uint8_t out[ENC_HEADER_BYTES + 8 + ENC_PREFIX_BYTES]) {
    memcpy(out, hdr, ENC_HEADER_BYTES);
    put_u64be(out + ENC_HEADER_BYTES, index);
    memcpy(out + ENC_HEADER_BYTES + 8, prefix, ENC_PREFIX_BYTES);
}

/* ---- key loading ---- */

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int enc_key_load(const char *path, uint8_t key[ENC_KEY_BYTES]) {
    if (!path || !key) { errno = EINVAL; return -1; }

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;

    uint8_t raw[80];
    ssize_t n = read_upto(fd, raw, sizeof raw);
    int saved = errno;
    close(fd);
    if (n < 0) { errno = saved; return -1; }

    /* Exact-length raw key wins BEFORE any trimming: 4 of 256 possible final
     * bytes are whitespace, so trimming first rejects ~1.6% of random keys. */
    if (n == ENC_KEY_BYTES) { memcpy(key, raw, ENC_KEY_BYTES); return 0; }

    /* trim trailing whitespace so a hex file written by `echo` still loads */
    while (n > 0 && (raw[n - 1] == '\n' || raw[n - 1] == '\r' ||
                     raw[n - 1] == ' '  || raw[n - 1] == '\t')) n--;

    if (n == ENC_KEY_BYTES) { memcpy(key, raw, ENC_KEY_BYTES); return 0; }

    if (n == ENC_KEY_BYTES * 2) {
        for (int i = 0; i < ENC_KEY_BYTES; i++) {
            int hi = hexval(raw[2 * i]), lo = hexval(raw[2 * i + 1]);
            if (hi < 0 || lo < 0) { errno = EINVAL; return -1; }
            key[i] = (uint8_t)((hi << 4) | lo);
        }
        return 0;
    }

    errno = EINVAL;
    return -1;
}

/* ---- sink ---- */

static int seal_chunk(enc_sink_t *s, int final) {
    uint8_t prefix[ENC_PREFIX_BYTES] = { 0 };
    put_u32le(prefix, s->buf_len);
    prefix[4] = final ? CFLAG_FINAL : 0;

    uint8_t aad[ENC_HEADER_BYTES + 8 + ENC_PREFIX_BYTES];
    build_aad(s->hdr, s->index, prefix, aad);

    uint8_t nonce[ENC_NONCE_BYTES];
    chunk_nonce(s->base_nonce, s->index, nonce);

    uint8_t *ct = s->ct;               /* preallocated: sealing runs ~4x/s */
    int ok = 0, outl = 0, tmpl = 0;
    uint8_t tag[ENC_TAG_BYTES];

    if (EVP_EncryptInit_ex(s->ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) == 1 &&
        EVP_CIPHER_CTX_ctrl(s->ctx, EVP_CTRL_GCM_SET_IVLEN,
                            ENC_NONCE_BYTES, NULL) == 1 &&
        EVP_EncryptInit_ex(s->ctx, NULL, NULL, s->key, nonce) == 1 &&
        EVP_EncryptUpdate(s->ctx, NULL, &tmpl, aad, (int)sizeof aad) == 1 &&
        (s->buf_len == 0 ||
         EVP_EncryptUpdate(s->ctx, ct, &outl, s->buf, (int)s->buf_len) == 1) &&
        EVP_EncryptFinal_ex(s->ctx, ct + outl, &tmpl) == 1 &&
        EVP_CIPHER_CTX_ctrl(s->ctx, EVP_CTRL_GCM_GET_TAG,
                            ENC_TAG_BYTES, tag) == 1)
        ok = 1;

    if (ok) {
        ok = write_all(s->fd, prefix, sizeof prefix) == 0 &&
             (s->buf_len == 0 || write_all(s->fd, ct, s->buf_len) == 0) &&
             write_all(s->fd, tag, sizeof tag) == 0;
    }

    if (!ok) { s->err = 1; return -1; }

    s->cipher_bytes += ENC_PREFIX_BYTES + s->buf_len + ENC_TAG_BYTES;
    s->index++;
    s->buf_len = 0;
    return 0;
}

enc_sink_t *enc_sink_open(int fd, const uint8_t key[ENC_KEY_BYTES],
                          const char *device_id, uint32_t chunk_bytes) {
    if (fd < 0 || !key) { errno = EINVAL; return NULL; }

    if (chunk_bytes == 0)             chunk_bytes = ENC_CHUNK_DEFAULT;
    else if (chunk_bytes < ENC_CHUNK_MIN) chunk_bytes = ENC_CHUNK_MIN;
    else if (chunk_bytes > ENC_CHUNK_MAX) chunk_bytes = ENC_CHUNK_MAX;

    enc_sink_t *s = calloc(1, sizeof *s);
    if (!s) return NULL;

    s->fd = fd;
    s->chunk_bytes = chunk_bytes;
    memcpy(s->key, key, ENC_KEY_BYTES);

    s->buf = malloc(chunk_bytes);
    s->ct  = malloc(chunk_bytes);
    s->ctx = EVP_CIPHER_CTX_new();
    if (!s->buf || !s->ct || !s->ctx) goto fail;

    if (RAND_bytes(s->base_nonce, ENC_NONCE_BYTES) != 1) { errno = EIO; goto fail; }

    s->hdr[0] = ENC_MAGIC0; s->hdr[1] = ENC_MAGIC1;
    s->hdr[2] = ENC_MAGIC2; s->hdr[3] = ENC_MAGIC3;
    put_u16le(s->hdr + 4, ENC_VERSION);
    put_u16le(s->hdr + 6, ENC_HEADER_BYTES);
    put_u32le(s->hdr + 8, chunk_bytes);
    s->hdr[0x0C] = ENC_ALG_AES_256_GCM;
    memcpy(s->hdr + 0x10, s->base_nonce, ENC_NONCE_BYTES);
    if (efr_key_id_bytes(key, ENC_KEY_BYTES, s->hdr + 0x1C) != 0) { errno = EIO; goto fail; }
    if (device_id) {
        size_t n = strlen(device_id);
        if (n > ENC_DEVICE_ID_MAX) n = ENC_DEVICE_ID_MAX;
        memcpy(s->hdr + 0x20, device_id, n);
    }

    if (write_all(fd, s->hdr, ENC_HEADER_BYTES) != 0) goto fail;
    s->cipher_bytes = ENC_HEADER_BYTES;
    return s;

fail: {
        int saved = errno;
        if (s->ctx) EVP_CIPHER_CTX_free(s->ctx);
        free(s->buf);
        free(s->ct);
        free(s);
        errno = saved;
        return NULL;
    }
}

ssize_t enc_sink_write(enc_sink_t *s, const void *buf, size_t len) {
    if (!s || (!buf && len)) { errno = EINVAL; return -1; }
    if (s->err) { errno = EIO; return -1; }

    const uint8_t *p = buf;
    size_t left = len;
    while (left) {
        uint32_t space = s->chunk_bytes - s->buf_len;
        uint32_t take  = left < space ? (uint32_t)left : space;
        memcpy(s->buf + s->buf_len, p, take);
        s->buf_len += take;
        p += take; left -= take;
        if (s->buf_len == s->chunk_bytes && seal_chunk(s, 0) != 0) return -1;
    }
    s->plain_bytes += len;
    return (ssize_t)len;
}

static void sink_free(enc_sink_t **sp) {
    enc_sink_t *s = *sp;
    if (s->ctx) EVP_CIPHER_CTX_free(s->ctx);
    free(s->buf);
    free(s->ct);
    free(s);
    *sp = NULL;
}

int enc_sink_flush(enc_sink_t *s) {
    if (!s) { errno = EINVAL; return -1; }
    if (s->err) { errno = EIO; return -1; }
    if (s->buf_len == 0) return 0;
    return seal_chunk(s, 0);
}

int enc_sink_finish(enc_sink_t **sp, uint64_t *out_cipher_bytes) {
    if (!sp || !*sp) { errno = EINVAL; return -1; }
    enc_sink_t *s = *sp;
    int rc = s->err ? -1 : seal_chunk(s, 1);   /* zero-length final is fine */
    if (out_cipher_bytes) *out_cipher_bytes = s->cipher_bytes;
    sink_free(sp);
    return rc;
}

void enc_sink_abort(enc_sink_t **sp) {
    if (!sp || !*sp) return;
    sink_free(sp);
}

int      enc_sink_err         (const enc_sink_t *s) { return s ? s->err : 1; }
uint64_t enc_sink_plain_bytes (const enc_sink_t *s) { return s ? s->plain_bytes : 0; }
uint64_t enc_sink_cipher_bytes(const enc_sink_t *s) { return s ? s->cipher_bytes : 0; }

/* ---- decrypt ---- */

int enc_decrypt_fd(int in_fd, int out_fd, const uint8_t key[ENC_KEY_BYTES],
                   struct enc_dec_info *info) {
    if (in_fd < 0 || out_fd < 0 || !key) { errno = EINVAL; return -1; }

    struct enc_dec_info local;
    if (!info) info = &local;
    memset(info, 0, sizeof *info);

    uint8_t hdr[ENC_HEADER_BYTES];
    if (read_upto(in_fd, hdr, ENC_HEADER_BYTES) != ENC_HEADER_BYTES) {
        errno = EINVAL; return -1;
    }
    if (hdr[0] != ENC_MAGIC0 || hdr[1] != ENC_MAGIC1 ||
        hdr[2] != ENC_MAGIC2 || hdr[3] != ENC_MAGIC3 ||
        get_u16le(hdr + 4) != ENC_VERSION ||
        get_u16le(hdr + 6) != ENC_HEADER_BYTES) { errno = EINVAL; return -1; }

    uint32_t chunk_bytes = get_u32le(hdr + 8);
    if (chunk_bytes < ENC_CHUNK_MIN || chunk_bytes > ENC_CHUNK_MAX) {
        errno = EINVAL; return -1;
    }
    uint8_t base_nonce[ENC_NONCE_BYTES];
    memcpy(base_nonce, hdr + 0x10, ENC_NONCE_BYTES);
    memcpy(info->device_id, hdr + 0x20, ENC_DEVICE_ID_MAX);
    info->device_id[ENC_DEVICE_ID_MAX] = '\0';
    info->chunk_bytes = chunk_bytes;
    info->algorithm   = hdr[0x0C];

    /* Fill info before either early return, so a caller can say WHICH key the
     * file wants even when it cannot decrypt it. */
    uint8_t file_id[ENC_KEY_ID_BYTES];
    memcpy(file_id, hdr + 0x1C, ENC_KEY_ID_BYTES);
    int have_id = 0;
    for (size_t i = 0; i < ENC_KEY_ID_BYTES; i++) if (file_id[i]) { have_id = 1; break; }
    if (have_id) efr_key_id_hex(file_id, info->key_id);

    if (info->algorithm != ENC_ALG_UNSPECIFIED &&
        info->algorithm != ENC_ALG_AES_256_GCM) { errno = ENOTSUP; return -1; }

    /* A key_id mismatch is the wrong key, which would otherwise surface as every
     * chunk failing its tag, which is true but unhelpful. */
    if (have_id) {
        uint8_t want[ENC_KEY_ID_BYTES];
        if (efr_key_id_bytes(key, ENC_KEY_BYTES, want) != 0) { errno = EIO; return -1; }
        if (memcmp(want, file_id, ENC_KEY_ID_BYTES) != 0) { errno = EKEYREJECTED; return -1; }
    }

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    uint8_t *ct = malloc(chunk_bytes);
    uint8_t *pt = malloc(chunk_bytes);
    if (!ctx || !ct || !pt) { errno = ENOMEM; goto io_fail; }

    int rc = 0;                       /* 0 clean, 1 truncated/tag failure */
    uint64_t index = 0;

    for (;;) {
        uint8_t prefix[ENC_PREFIX_BYTES];
        ssize_t n = read_upto(in_fd, prefix, ENC_PREFIX_BYTES);
        if (n < 0) goto io_fail;
        if (n == 0) { rc = 1; break; }               /* no end marker */
        if (n != ENC_PREFIX_BYTES) { rc = 1; break; } /* torn prefix */

        uint32_t plain_len = get_u32le(prefix);
        int final = (prefix[4] & CFLAG_FINAL) != 0;
        if (plain_len > chunk_bytes) { rc = 1; break; }

        if (plain_len) {
            n = read_upto(in_fd, ct, plain_len);
            if (n < 0) goto io_fail;
            if ((uint32_t)n != plain_len) { rc = 1; break; }
        }
        uint8_t tag[ENC_TAG_BYTES];
        n = read_upto(in_fd, tag, ENC_TAG_BYTES);
        if (n < 0) goto io_fail;
        if (n != ENC_TAG_BYTES) { rc = 1; break; }

        uint8_t aad[ENC_HEADER_BYTES + 8 + ENC_PREFIX_BYTES];
        build_aad(hdr, index, prefix, aad);
        uint8_t nonce[ENC_NONCE_BYTES];
        chunk_nonce(base_nonce, index, nonce);

        int outl = 0, tmpl = 0;
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1 ||
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN,
                                ENC_NONCE_BYTES, NULL) != 1 ||
            EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1 ||
            EVP_DecryptUpdate(ctx, NULL, &tmpl, aad, (int)sizeof aad) != 1 ||
            (plain_len &&
             EVP_DecryptUpdate(ctx, pt, &outl, ct, (int)plain_len) != 1) ||
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                ENC_TAG_BYTES, tag) != 1 ||
            EVP_DecryptFinal_ex(ctx, pt + outl, &tmpl) != 1) {
            rc = 1; break;                            /* tag failed */
        }

        if (plain_len && write_all(out_fd, pt, plain_len) != 0) goto io_fail;

        info->chunks++;
        info->plain_bytes += plain_len;
        index++;

        if (final) { info->clean_eof = 1; break; }
    }

    EVP_CIPHER_CTX_free(ctx);
    free(ct); free(pt);
    return rc;

io_fail: {
        int saved = errno;
        if (ctx) EVP_CIPHER_CTX_free(ctx);
        free(ct); free(pt);
        errno = saved;
        return -1;
    }
}
