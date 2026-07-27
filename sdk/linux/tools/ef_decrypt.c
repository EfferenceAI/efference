/* ef-decrypt — turn an encrypted recording back into the plain container.
 *
 * The whole format lives in rec/enc_sink.{c,h}; this is only a CLI around
 * enc_decrypt_fd(), so there is exactly one implementation of the format and the
 * device and host can never disagree about it.
 *
 * Exit codes mirror enc_decrypt_fd's three outcomes, because "truncated" is a
 * normal result for a recording cut short by power loss and must not look like
 * corruption:
 *   0  clean: the end marker was present
 *   1  truncated or a chunk failed its tag; output up to that point is valid
 *   2  unusable input (bad header, wrong key from the first chunk, I/O)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include "enc_sink.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <in.enc> <key-file> <out>\n"
                "  key-file: 32 raw bytes or 64 hex chars\n", argv[0]);
        return 2;
    }

    uint8_t key[ENC_KEY_BYTES];
    if (enc_key_load(argv[2], key) != 0) {
        fprintf(stderr, "ef-decrypt: key %s: %s\n", argv[2], strerror(errno));
        return 2;
    }

    int in = open(argv[1], O_RDONLY);
    if (in < 0) { fprintf(stderr, "ef-decrypt: %s: %s\n", argv[1], strerror(errno)); return 2; }
    int out = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        fprintf(stderr, "ef-decrypt: %s: %s\n", argv[3], strerror(errno));
        close(in);
        return 2;
    }

    struct enc_dec_info info;
    int rc = enc_decrypt_fd(in, out, key, &info);
    close(in);
    close(out);

    if (rc < 0) {
        int saved = errno;
        /* Leave nothing behind: a 0-byte .mcap reads as a successful decrypt of an
         * empty recording, and partial plaintext is worse. */
        unlink(argv[3]);
        errno = saved;
        /* Name the two failures a user can act on; anything else is corruption. */
        if (errno == EKEYREJECTED)
            /* The header is unauthenticated at this point (the chunk tags are what
             * bind it), so a modified key_id is indistinguishable from the wrong
             * key. Say both rather than asserting a key that may never have
             * existed. */
            fprintf(stderr, "ef-decrypt: header names key %s: wrong key, "
                            "or a modified header\n", info.key_id);
        else if (errno == ENOTSUP)
            fprintf(stderr, "ef-decrypt: algorithm %u is not supported by this build\n",
                    info.algorithm);
        else
            fprintf(stderr, "ef-decrypt: not a valid encrypted recording (%s)\n", strerror(errno));
        return 2;
    }

    fprintf(stderr, "device_id : %s\n", info.device_id[0] ? info.device_id : "(unset)");
    fprintf(stderr, "key_id    : %s\n", info.key_id[0] ? info.key_id : "(unnamed, pre-key_id file)");
    fprintf(stderr, "algorithm : AES-256-GCM (id %u)\n", info.algorithm);
    fprintf(stderr, "chunks    : %llu (%u B each)\n",
            (unsigned long long)info.chunks, info.chunk_bytes);
    fprintf(stderr, "plaintext : %llu bytes -> %s\n",
            (unsigned long long)info.plain_bytes, argv[3]);
    if (!info.clean_eof)
        fprintf(stderr, "WARNING   : no end marker; file was truncated. "
                        "Everything above decrypted and verified.\n");
    return rc == 0 ? 0 : 1;
}
