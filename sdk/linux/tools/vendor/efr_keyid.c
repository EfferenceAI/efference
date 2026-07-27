/* efr_keyid.c — see efr_keyid.h. OpenSSL EVP only, matching the rest of the tree. */
#include "efr_keyid.h"

#include <string.h>

#include <openssl/evp.h>

int efr_key_id_bytes(const uint8_t *key, size_t key_len, uint8_t out[EFR_KEY_ID_BYTES]) {
    uint8_t d[EVP_MAX_MD_SIZE];
    unsigned int dlen = 0;
    if (!key || !out || key_len == 0) return -1;
    if (EVP_Digest(key, key_len, d, &dlen, EVP_sha256(), NULL) != 1 ||
        dlen < EFR_KEY_ID_BYTES)
        return -1;
    memcpy(out, d, EFR_KEY_ID_BYTES);
    return 0;
}

void efr_key_id_hex(const uint8_t id[EFR_KEY_ID_BYTES], char out[EFR_KEY_ID_STRLEN]) {
    static const char h[] = "0123456789abcdef";
    for (size_t i = 0; i < EFR_KEY_ID_BYTES; i++) {
        out[2*i]   = h[id[i] >> 4];
        out[2*i+1] = h[id[i] & 0xf];
    }
    out[EFR_KEY_ID_BYTES * 2] = '\0';
}

int efr_key_id(const uint8_t *key, size_t key_len, char out[EFR_KEY_ID_STRLEN]) {
    uint8_t id[EFR_KEY_ID_BYTES];
    if (!out || efr_key_id_bytes(key, key_len, id) != 0) return -1;
    efr_key_id_hex(id, out);
    return 0;
}
