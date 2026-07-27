#ifndef EFR_KEYID_H
#define EFR_KEYID_H

#include <stddef.h>
#include <stdint.h>

/* Short public name for an encryption key: hex of the first EFR_KEY_ID_BYTES of
 * SHA-256(key). Not a secret; it identifies a key without revealing it.
 *
 * Lives here because two binaries must produce the identical value or the system
 * quietly stops making sense: efference-capture stamps it into every encrypted
 * file's header, and efference-sdk-endpoint reports it on GetDeviceInformation
 * and matches it in DeleteEncryptionKey. Same repo, but separate programs, so a
 * second copy would drift with nothing failing to build. */

#define EFR_KEY_ID_BYTES  4
#define EFR_KEY_ID_STRLEN (EFR_KEY_ID_BYTES * 2 + 1)   /* hex + NUL */

/* Raw id bytes for a 32-byte key. 0 / -1. */
int efr_key_id_bytes(const uint8_t *key, size_t key_len, uint8_t out[EFR_KEY_ID_BYTES]);

/* Hex form of raw id bytes. Split from the above so a caller that already has
 * the bytes (a file header) formats them without re-hashing a key it lacks. */
void efr_key_id_hex(const uint8_t id[EFR_KEY_ID_BYTES], char out[EFR_KEY_ID_STRLEN]);

/* Hex id straight from a key. 0 / -1. */
int efr_key_id(const uint8_t *key, size_t key_len, char out[EFR_KEY_ID_STRLEN]);

#endif /* EFR_KEYID_H */
