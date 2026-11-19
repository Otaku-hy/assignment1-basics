import torch
import torch.nn as nn
import math
from einops import rearrange, einsum, reduce
import torch.cuda.nvtx as nvtx

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype = dtype, device= device))

        sigma = math.sqrt( 2 / (in_features + out_features) )
        nn.init.trunc_normal_( self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)
        #return einsum(x, self.weight, "... in, out in -> ... out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(num_embeddings,embedding_dim,dtype=dtype,device=device))

        sigma = 1
        nn.init.trunc_normal_(self.weight,0,1,-3 * sigma,3*sigma)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
        
class RMS_Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    @nvtx.range("rms_norm")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = 1.0 / torch.sqrt(reduce(x * x, "... seq dm -> ... seq 1", "mean") + self.eps)
        result = x * rms * self.weight

        return result.to(in_dtype)

# ablation four: silu
class SiLU_FFN(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        return self.w2(y * torch.sigmoid(y))

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype) #gate
        self.w3 = Linear(d_model, d_ff, device, dtype) #value
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        g = x1 * torch.sigmoid(x1)
        v = self.w3(x)

        return self.w2(g*v)

class RotrayPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, dtype=None, device=None):
        super().__init__()

        freq = theta ** -((2 * torch.arange(1,d_k//2+1,1,device=device) - 2)/ d_k)
        index = torch.arange(0,max_seq_len,1,dtype=torch.float32, device=device)

        theta_map = einsum(index, freq, "i, d -> i d")
        cosine = torch.cos(theta_map)
        sine = torch.sin(theta_map)

        cosine = torch.repeat_interleave(cosine,2,-1)
        sine = torch.repeat_interleave(sine,2,-1)
        sine = sine * ((-1) ** torch.arange(1,d_k+1,1,device=device))

        self.register_buffer("cos",cosine.to(dtype=dtype),False)
        self.register_buffer("sin",sine.to(dtype=dtype), False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        index = torch.arange(x.shape[-1],device=x.device) ^ 1
        x_prime = x[...,index]

        cosine = self.cos[token_positions]
        sine = self.sin[token_positions]

        return (x * cosine + x_prime * sine).to(dtype=x.dtype)

@nvtx.range("softmax")
def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    # use fp32 for exp computation
    in_type = x.dtype
    x = x.to(torch.float32)

    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)

    result = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

    return result.to(in_type)

def scaled_dot_product_attention(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:
    dim = q.shape[-1]
    with nvtx.range("attn-qkT"):
        score = einsum(q,k, "... queries d_k, ... keys d_k -> ... queries keys")
        score /= math.sqrt(dim)

    with nvtx.range("attn-softmax"):
        if mask == None:
            logits = softmax(score, -1)
        else :
            score_masked = torch.where(mask, score, float("-inf"))
            logits = softmax(score_masked, -1)

    with nvtx.range("attn-sv"):
        result = einsum(logits, v, "... queries keys, ... keys d_v -> ... queries d_v")
    return result

class Multihead_SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope : RotrayPositionalEmbedding = None, device = None, dtype = None):
        super().__init__()

        self.num_heads = num_heads
        d_k = d_q = d_v = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * d_q, device, dtype)
        self.k_proj = Linear(d_model, num_heads * d_k, device ,dtype)
        self.v_proj = Linear(d_model, num_heads * d_v, device, dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device, dtype)

        #ablation three - NoPE
        #self.rope = None

        self.rope = rope


    def forward(self, x : torch.Tensor, token_position : torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        qs = rearrange(Q, "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads = self.num_heads)
        ks = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads = self.num_heads)
        vs = rearrange(V, "... seq (num_heads d_v) ->... num_heads seq d_v", num_heads = self.num_heads)

        if self.rope is not None:
            if token_position == None:
                token_position = torch.arange(0,x.shape[-2],1,device=x.device)
            token_position = rearrange(token_position, "... seq -> ... 1 seq")
            qs = self.rope(qs, token_position)
            ks = self.rope(ks, token_position)

        mask = torch.tril(torch.ones(qs.shape[-2],ks.shape[-2],device=x.device,dtype=bool),0)
        
        value = scaled_dot_product_attention(qs,ks,vs,mask)

        value = rearrange(value, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        return self.output_proj(value)

class GroupedQuery_SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_q_per_group:int, rope : RotrayPositionalEmbedding = None, device = None, dtype = None):
        super().__init__()

        self.num_heads = num_heads
        self.num_q_per_group = num_q_per_group

        d_k = d_q = d_v = d_model // num_heads
        k_heads = v_heads = num_heads // num_q_per_group

        self.q_proj = Linear(d_model, num_heads * d_q, device, dtype)
        self.k_proj = Linear(d_model, k_heads * d_k, device ,dtype)
        self.v_proj = Linear(d_model, v_heads * d_v, device, dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device, dtype)

        self.rope = rope

    def forward(self, x : torch.Tensor, token_position : torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        k_heads = v_heads = self.num_heads // self.num_q_per_group

        qs = rearrange(Q, "... seq (k_heads gq d_q) -> ... k_heads gq seq d_q", gq = self.num_q_per_group, k_heads = k_heads)
        ks = rearrange(K, "... seq (k_heads d_k) -> ... k_heads 1 seq d_k", k_heads = k_heads)
        vs = rearrange(V, "... seq (v_heads d_v) ->... v_heads 1 seq d_v", v_heads = v_heads)

        if self.rope is not None:
            if token_position == None:
                token_position = torch.arange(0,x.shape[-2],1,device=x.device)
            token_position = rearrange(token_position, "... seq -> ... 1 1 seq")
            qs = self.rope(qs, token_position)
            ks = self.rope(ks, token_position)

        mask = torch.tril(torch.ones(qs.shape[-2],ks.shape[-2],device=x.device,dtype=bool),0)
        
        value = scaled_dot_product_attention(qs,ks,vs,mask)

        value = rearrange(value, "... k_heads gq seq d_v -> ... seq (k_heads gq d_v)")
        return self.output_proj(value)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, num_q_per_group:int=None, rope:RotrayPositionalEmbedding = None, device = None, dtype = None):
        super().__init__()
        self.ln1 = RMS_Norm(d_model,device=device,dtype=dtype)
        if num_q_per_group is None:
            self.attn = Multihead_SelfAttention(d_model,num_heads,rope, device, dtype)
        else:
            self.attn = GroupedQuery_SelfAttention(d_model,num_heads, num_q_per_group,rope, device, dtype)
        self.ln2 = RMS_Norm(d_model,device=device,dtype=dtype)
        
        self.ffn = SwiGLU_FFN(d_model,d_ff,device,dtype)
        # ablation four - silu
        #self.ffn = SiLU_FFN(d_model,d_ff,device,dtype)

    @nvtx.range("transformer block")
    def forward(self, x : torch.Tensor, token_position : torch.Tensor=None) -> torch.Tensor:
        y = x +  self.attn(self.ln1(x),token_position)
        return y + self.ffn(self.ln2(y))

        #ablation one - no ln
        #y = x + self.attn(x, token_position)
        #eturn y + self.ffn(y)

        #ablation two - post ln
        #y = self.ln1(x + self.attn(x, token_position))
        #return self.ln2(y + self.ffn(y))
    
        

