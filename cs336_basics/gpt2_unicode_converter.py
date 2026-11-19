from functools import lru_cache

def gpt2_bytes_to_unicode() -> dict[int, str]:

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:

            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

gpt2_byte_encoder = gpt2_bytes_to_unicode()
gpt2_byte_decoder = { v : k for k,v in gpt2_byte_encoder.items()}

def token_bytes_to_unicode(token_bytes : bytes) -> str:
    return "".join([gpt2_byte_encoder[byte] for byte in token_bytes])

def token_unicode_to_bytes(token_unicode : str) -> bytes:
    return bytes([gpt2_byte_decoder[ch] for ch in token_unicode])

