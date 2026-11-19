import torch

from cs336_basics.transformer import Transformer
from cs336_basics.generate_text import generate_text, load_model
from cs336_basics.BPE_Tokenizer import BPE_Tokenizer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"   # 用当前机器的，不读 config 里的 "cuda"
    model, args = load_model("runs/80M/full", "ckpt_50000.pt", device)
    tokenizer = BPE_Tokenizer.from_files("data/owt_dict.json", "data/owt_merges.txt",
                                         ["<|endoftext|>"])
    print(generate_text(tokenizer, model, "Computer science is", max_length=200, temperature=0.8, top_p=0.9))
