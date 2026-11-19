import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.transformer_building_blocks import *
from cs336_basics.transformer_building_blocks import RotrayPositionalEmbedding as RoPE


class Transformer(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta: float, vocab_size:int, context_length:int, num_layers:int, num_q_per_group:int=None, device = None, dtype = None, grad_ckpt = False):
        super().__init__()

        d_k = d_model // num_heads
        self.grad_ckpt = grad_ckpt

        self.rope = RoPE(theta,d_k,context_length,dtype,device)

        self.layers = nn.Sequential(*[TransformerBlock(d_model,num_heads,d_ff,num_q_per_group,self.rope,device,dtype) for _ in range(num_layers)])

        self.token_embeddings = Embedding(vocab_size,d_model,device,dtype)

        self.ln_final = RMS_Norm(d_model,device=device,dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size,device, dtype)

    def forward(self, indices:torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(indices)
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                x = checkpoint(layer,x,use_reentrant=False)
            else:
                x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
