from cs336_basics.BPE_Tokenizer import train_BPE

import json
import time
from functools import lru_cache

@lru_cache
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

def token_bytes_to_unicode(token_bytes : bytes) -> str:
    return "".join([gpt2_byte_encoder[byte] for byte in token_bytes])

special_tokens = ["<|endoftext|>"]
vocab_size = 32000

begin = time.perf_counter()
vocab, merges = train_BPE(vocab_size, special_tokens, "data/owt_train.txt")
end = time.perf_counter()

print(end - begin)

print(max(vocab.values(), key=len))


vocab_map = {}
merges_map = {}

for idx, token_bytes in vocab.items():
    vocab_map[token_bytes_to_unicode(token_bytes)] = idx

with open("data/owt_dict.json","w") as file:
    json.dump(vocab_map,file,ensure_ascii=False)

with open("data/owt_dict.txt","w") as file:
    for left, right in merges:
        file.write(token_bytes_to_unicode(left) + " " + token_bytes_to_unicode(right) + "\n")

