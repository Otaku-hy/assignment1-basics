import torch
import argparse
import json
import os


from cs336_basics.transformer import Transformer
from cs336_basics.BPE_Tokenizer import BPE_Tokenizer

from cs336_basics.transformer_building_blocks import softmax


def load_model(run_dir: str, ckpt_name: str, device: str) -> tuple[Transformer, argparse.Namespace]:
    with open(os.path.join(run_dir, "config.json")) as f:
        args = argparse.Namespace(**json.load(f))

    model = Transformer(args.d_model, args.num_heads, args.d_ff, args.rope_theta, args.vocab_size,
                        args.context_length, args.num_layers, args.num_q_per_group, device, torch.float32)

    ckpt = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model, args

def generate_tokens(model:torch.nn.Module, prompt:torch.Tensor, max_length:int, temperature:float, top_p:float = 1) -> torch.Tensor:
    model.eval()

    generated = 0
    if temperature == 0:
        raise ValueError("temperature can not be 0")

    while generated < max_length:
        
        with torch.no_grad():
            logits = model(prompt)

        v = logits[...,-1,:]
        p = softmax(v/temperature,dim=-1)
        p, index = torch.sort(p,descending=True,dim=-1)
        exclusive = torch.cumsum(p,dim=-1) - p

        mask = exclusive < top_p

        p = torch.where(mask, p, 0)
        p /= torch.sum(p)

        pos = torch.multinomial(p,num_samples=1)
        token_id = torch.gather(index,dim=-1,index=pos)

        prompt = torch.cat([prompt,token_id],dim=-1)
        generated+=1
        if token_id == 256: break

    return prompt

def generate_text(tokenzier:BPE_Tokenizer, model:torch.nn.Module, prompt:str, max_length:int, temperature:float, top_p:float = 1) -> str:
    if prompt is None or prompt == "":
        prompt = "<|endoftext|>"

    device = next(model.parameters()).device
    token_list = tokenzier.encode(prompt)
    tokens = torch.tensor(token_list,device=device)

    tokens = generate_tokens(model,tokens,max_length,temperature,top_p)

    return tokenzier.decode(tokens.tolist())









    
