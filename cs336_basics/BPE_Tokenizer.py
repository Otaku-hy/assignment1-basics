from typing import BinaryIO
import regex as re
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cs336_basics.gpt2_unicode_converter as Converter
import json
from collections.abc import Iterable, Iterator

def update_bytes_list(bytes_list : list[bytes], pair : tuple[bytes, bytes]) -> list[bytes]:
    list_updated = []
    pair_bytes = pair[0] + pair[1]
    i = 0
    while i in range(len(bytes_list)):
        if i + 1 < len(bytes_list) and (bytes_list[i],bytes_list[i+1]) == pair:
            list_updated.append(pair_bytes)
            i = i + 2
        else:
            list_updated.append(bytes_list[i])
            i = i + 1

    return list_updated

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    if split_special_token == b"":
        split_special_token = b" "

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(b_start : int, b_end : int, input_path: str | os.PathLike, special_tokens : list) -> dict[str,int] :
    pretoken_dict = {}

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    chunk_size = b_end - b_start
    with open(input_path, "rb") as file:
        file.seek(b_start)
        chunk = file.read(chunk_size).decode("utf-8", errors = "ignore")

    slices = re.split("|".join(special_tokens), chunk) if len(special_tokens) > 0 else [chunk]

    for s in slices :
        it = re.finditer(PAT, s)
        for match in it:
            token = match.group()
            pretoken_dict[token] = pretoken_dict.get(token, 0) + 1 

    return pretoken_dict        
        
def train_BPE(desired_vocab_size : int, special_tokens : list, input_path: str | os.PathLike, num_workers : int = 20) -> tuple :
    #init vocab
    vocab = {}
    merge = []

    for i in range(256) :
        vocab[i] = bytes([i])

    i = len(vocab)
    for token in special_tokens:
        vocab[i] = token.encode("utf-8")
        i= i + 1
    vocab_size = len(vocab)

    with open(input_path, "rb") as file:
        split_token = special_tokens[0].encode("utf-8") if len(special_tokens) > 0 else b""
        boundaries = find_chunk_boundaries(file, desired_num_chunks= num_workers, split_special_token= split_token)

    translated_special_tokens = [re.escape(token) for token in special_tokens]
    translated_special_tokens = sorted(translated_special_tokens, key= lambda tokens : len(tokens), reverse=True)

    pretoken_dict = Counter()
    pretoken_vocab_seq = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_pretoken_dicts = {executor.submit(pre_tokenization, b_start, b_end, input_path, translated_special_tokens) for b_start, b_end in zip(boundaries[:-1],boundaries[1:])}
        for future_dict in future_pretoken_dicts:
            thread_dict = future_dict.result()
            pretoken_dict.update(thread_dict)

    for token in pretoken_dict:
        word_encoded = token.encode("utf-8")
        pretoken_vocab_seq[token] = [bytes([b]) for b in word_encoded]

    #now we get the dictionary, do merging
    merging_pair = {}
    entry_table = {}

    for pretoken in pretoken_dict:
        lists = pretoken_vocab_seq[pretoken]
        pairs = zip(lists[:-1],lists[1:])
        for pair in pairs:
            count = merging_pair.get(pair, 0)
            count = count + pretoken_dict[pretoken]
            merging_pair[pair] = count
            if pair not in entry_table:
                entry_table[pair] = set()
            entry_table[pair].add(pretoken)

    while merging_pair and vocab_size < desired_vocab_size :
        pair, _ = max(merging_pair.items(), key=lambda item: (item[1], item[0]))
        pair_bytes = pair[0] + pair[1]
        merge.append(pair)

        vocab[vocab_size] = pair_bytes
        vocab_size = vocab_size + 1
        
        entries = entry_table.pop(pair)
        merging_pair.pop(pair)
        for entry in entries:
            word = entry
            count = pretoken_dict[word]
            list_odd = pretoken_vocab_seq[word]

            list_updated = update_bytes_list(list_odd, pair)
            pretoken_vocab_seq[word] = list_updated

            pair_odd = zip(list_odd[:-1],list_odd[1:])
            pair_new = zip(list_updated[:-1],list_updated[1:])

            for pair_it in pair_odd:
                if pair_it == pair: continue
                merging_pair[pair_it] = merging_pair.get(pair_it, 0) - count
                entry_table[pair_it].discard(word)

            for pair_it in pair_new:
                merging_pair[pair_it] = merging_pair.get(pair_it, 0) + count
                if pair_it not in entry_table:
                    entry_table[pair_it] = set()
                entry_table[pair_it].add(word)

            for pair_it in zip(list_odd[:-1],list_odd[1:]):
                if pair_it == pair: continue
                if merging_pair.get(pair_it, 0) <= 0 : 
                    merging_pair.pop(pair_it, None)
                    entry_table.pop(pair_it, None)
                        
    return vocab, merge

class BPE_Tokenizer:
    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes,bytes]], special_tokens : list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if special_tokens == None else special_tokens
        translated_special_tokens = [re.escape(token) for token in self.special_tokens]
        self.translated_special_tokens = sorted(translated_special_tokens, key = len, reverse=True)

        vocab_size = len(self.vocab)
        tokens_bytes = [token.encode("utf-8") for token in self.special_tokens ]
        for token in tokens_bytes:
            if token not in self.vocab.values():
                self.vocab[vocab_size] = token
                vocab_size = vocab_size + 1

        self.inverse_vocab = { v : k for k,v in self.vocab.items()}
        self.merges_idx = { merge : i for i, merge in enumerate(self.merges) }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(merges_filepath, encoding="utf-8") as f:
            merges_unicode = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes(Converter.gpt2_byte_decoder[token] for token in merge_token_1),
                    bytes(Converter.gpt2_byte_decoder[token] for token in merge_token_2),
                )
                for merge_token_1, merge_token_2 in merges_unicode
            ]

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_unicode = json.load(f)
            vocab = {
                vocab_index: bytes(Converter.gpt2_byte_decoder[token] for token in vocab_item)
                for vocab_item, vocab_index in vocab_unicode.items()
            }

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        #pre tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_encoded = []
        
        slices = re.split("(" + "|".join(self.translated_special_tokens) + ")", text) if len(self.translated_special_tokens) > 0 else [text]
        for slc in slices:
            if slc in self.special_tokens:
                text_encoded.append(self.inverse_vocab[slc.encode("utf-8")])
            else:
                tokens = re.findall(PAT, slc)
                for token in tokens:
                    token_bytes = token.encode("utf-8")
                    bytes_list =  [bytes([b]) for b in token_bytes]
                    b_merge_exist = True
                    while len(bytes_list) > 1 and b_merge_exist :
                        merge_pairs = zip(bytes_list[:-1],bytes_list[1:])
                        merge_idx = len(self.merges)
                        for pair in merge_pairs:
                            merge_idx = min(merge_idx, self.merges_idx.get(pair,len(self.merges)))
                        if merge_idx == len(self.merges):
                            b_merge_exist = False
                        else:
                            bytes_list = update_bytes_list(bytes_list,self.merges[merge_idx])
                    token_encoded = [self.inverse_vocab[b] for b in bytes_list]
                    text_encoded.extend(token_encoded)

        return text_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for str in iterable:
            text_encoded = self.encode(str)
            yield from text_encoded


    def decode(self, ids : list[int]) -> str:
        text_bytes = b"".join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8", errors="replace")