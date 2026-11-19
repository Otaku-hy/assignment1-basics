import torch
from torch import nn
from einops import einsum, rearrange, reduce 

import numpy as np
import numpy.typing as npt

from collections.abc import Callable, Iterable
from typing import Optional
import math
import os
import typing

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration:int, out: str|os.PathLike|typing.BinaryIO|typing.IO):
    model_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()

    ckpt = {"model":model_dict,"optim":optimizer_dict,"it":iteration}

    torch.save(ckpt,out)

def load_checkpoint(src: str|os.PathLike|typing.BinaryIO|typing.IO, model: torch.nn.Module, optimizer:torch.optim.Optimizer) -> int:
    ckpt = torch.load(src)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])

    return ckpt["it"]

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = rearrange(logits, "... vocab_size -> (...) vocab_size")
    targets = rearrange(targets, "... -> (...) 1")

    logits = logits - torch.amax(logits,-1,keepdim=True)
    # fp32 
    logits = logits.to(torch.float32)
    z = torch.sum(torch.exp(logits),-1)
    u = torch.gather(logits,-1,targets)
    u = rearrange(u, "... 1 -> ...")

    loss = torch.log(z) - u
    return torch.mean(loss)

def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, targets))

def lr_cosine_schedule(t:int,alpha_max:float,alpha_min:float,Tw:int,Tc:int):
    if t < Tw: return t / Tw * alpha_max
    if t > Tc: return alpha_min
    return alpha_min + 0.5 * (1 + math.cos((t-Tw)/(Tc-Tw)*math.pi)) * (alpha_max - alpha_min)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = list(parameters)
    l2 = 0
    for p in parameters:
        if p.grad is None: continue
        l2 += torch.sum(p.grad ** 2)

    l2_norm = torch.sqrt(l2)
    if l2_norm < max_l2_norm: return
    for p in parameters:
        if p.grad is None: continue
        p.grad *= max_l2_norm / (l2_norm + 1e-6)
    
def load_data(dataset:npt.NDArray, batch_size:int, context_length:int, device:str) -> tuple[torch.Tensor,torch.Tensor]:
    rand_pos = np.random.randint(0,len(dataset)-context_length,batch_size)
    rand_pos = rearrange(rand_pos, "batch_size -> batch_size 1")
    rand_pos = rand_pos + np.arange(context_length)

    x_np = dataset[rand_pos]
    target_np = dataset[rand_pos + 1]
    x_np = x_np.astype(np.int64)
    target_np = target_np.astype(np.int64)

    x = torch.from_numpy(x_np)
    x = x.to(device)
    target = torch.from_numpy(target_np)
    target = target.to(device)

    return x,target
    
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.defaults = {"lr": lr}
        super().__init__(params,self.defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t+1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9,0.95), weight_decay=0.01, eps=1e-8):
        self.defaults = {"lr":lr, "betas":betas, "weight_decay":weight_decay, "eps":eps}
        super().__init__(params, self.defaults)

    def step(self, closure = None):
        for group in self.param_groups:
            #get hyper params for the group
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                dtype = p.data.dtype
                g = p.grad.float()
                
                state = self.state[p]
                if dtype == torch.bfloat16:
                    weight = state["weight"] if "weight" in state else p.data.to(dtype=torch.float32)
                else:
                    weight = p.data

                t = state.get("t",1)
                m = state["m"] if "m" in state else torch.zeros_like(p,dtype=torch.float32)
                v = state["v"] if "v" in state else torch.zeros_like(p,dtype=torch.float32)

                #decay first
                weight -= (lr * weight_decay) * weight
                     
                beta1_t = betas[0] ** t
                beta2_t = betas[1] ** t
                alpha_t = lr * math.sqrt(1-beta2_t)/(1-beta1_t)
                
                m = betas[0] * m + (1-betas[0]) * g
                v = betas[1] * v + (1-betas[1]) * (g ** 2)
                weight -= (alpha_t * m / (torch.sqrt(v) +eps))
                p.data.copy_(weight)

                state["t"] = t+1
                state["m"] = m
                state["v"] = v
                if dtype == torch.bfloat16:
                    state["weight"] = weight
        return None

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]

        for i, p in zip(saved_ids, params):
            if i not in state_dict["state"]:
                continue
            for k, val in state_dict["state"][i].items():
                if torch.is_tensor(val):    
                    self.state[p][k] = val.to(device=p.device)


