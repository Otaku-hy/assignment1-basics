import numpy as np

from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.BPE_Tokenizer import find_chunk_boundaries

def tokenize_dataset(vocab_path:str, merges_path:str, dataset_path:str, offset:int, size:int):
    tokenizer = BPE_Tokenizer.from_files(vocab_path,merges_path,["<|endoftext|>"])

    with open(dataset_path,"rb") as file:
        file.seek(offset)
        text_bytes = file.read(size)
        text = text_bytes.decode("utf-8")

        tokens = tokenizer.encode(text)
        return np.array(tokens, dtype=np.uint16)


if __name__ == "__main__":
    vocab_path = "data/owt_dict.json"
    merges_path = "data/owt_merges.txt"
    dataset_path = "data/owt_valid.txt"
    output_path = "data/owt_valid_token.bin"

    with open(dataset_path,"rb") as file:
        boundaries = find_chunk_boundaries(file,200,b"<|endoftext|>")

    with ProcessPoolExecutor(max_workers=20) as executor, open(output_path, "wb") as file:
            future_tokens = [executor.submit(tokenize_dataset,vocab_path,merges_path,dataset_path,b_start,b_end-b_start) for b_start, b_end in zip(boundaries[:-1],boundaries[1:])]
            for f in future_tokens:
                tokens = f.result()
                tokens.tofile(file)