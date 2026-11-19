import torch
import triton
import triton.language as tl

from einops import reduce, rearrange, einsum

class flash_attention_torch_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal=False):
        SEQ_Q = Q.shape[-2]
        SEQ_K = K.shape[-2]
        
        q_shape = Q.shape
        k_shape = K.shape
        v_shape = V.shape

        Q = rearrange(Q, "... seq_q d -> (...) seq_q d")
        K = rearrange(K, "... seq_k d -> (...) seq_k d")
        V = rearrange(V, "... seq_v d -> (...) seq_v d")

        ctx.q_shape = q_shape
        ctx.k_shape = k_shape
        ctx.v_shape = v_shape
        ctx.TILE_SIZE_Q = 16
        ctx.TILE_SIZE_K = 16
        ctx.is_causal = is_causal

        O = torch.zeros_like(Q,dtype=torch.float32)
        L = torch.zeros((Q.shape[0], SEQ_Q),device=Q.device,dtype=torch.float32)

        d = Q.shape[-1]

        for iter_q in range((SEQ_Q + ctx.TILE_SIZE_Q - 1) // ctx.TILE_SIZE_Q):
            Qi = Q[...,iter_q * ctx.TILE_SIZE_Q:(iter_q + 1) * ctx.TILE_SIZE_Q,:]
            mi = torch.full(Qi.shape[:-1], float("-inf"), dtype=torch.float32, device=Q.device)
            li = torch.zeros(Qi.shape[:-1], dtype=torch.float32, device=Q.device)
            Oi = torch.zeros(Qi.shape, dtype=torch.float32, device=Q.device)

            for iter_k in range((SEQ_K + ctx.TILE_SIZE_K - 1) // ctx.TILE_SIZE_K):
                Ki = K[...,iter_k * ctx.TILE_SIZE_K:(iter_k + 1) * ctx.TILE_SIZE_K,:]
                Vi = V[...,iter_k * ctx.TILE_SIZE_K:(iter_k + 1) * ctx.TILE_SIZE_K,:]

                Si = torch.matmul(Qi,Ki.transpose(-2,-1)).float() / (d ** 0.5)
                mi_prim = mi # m_i_j-1
                mi = torch.maximum(mi, torch.amax(Si,dim=-1)) #m_i_j
                Pi = torch.exp(Si - mi[...,None])
                li = li * torch.exp(mi_prim - mi) + torch.sum(Pi,dim=-1)
                Oi = Oi * (mi_prim[...,None] - mi[...,None]).exp() + torch.matmul(Pi,Vi)

            Oi = Oi * (1.0 / li[...,None])
            li = mi + torch.log(li)
            start = iter_q * ctx.TILE_SIZE_Q
            end = start + Qi.shape[-2]          # 用实际切出来的大小，最后一个 tile 不满时也对
            O[..., start:end, :] = Oi
            L[..., start:end] = li

        O = O.to(dtype=Q.dtype)
        ctx.save_for_backward(Q,K,V,O,L)

        return O.view(q_shape)

    @staticmethod
    def backward(ctx, dO):
        dO = rearrange(dO,"... seq_q d -> (...) seq_q d")
        Q,K,V,O,L = ctx.saved_tensors
        scale = 1.0 / (Q.shape[-1] ** 0.5)
        D = torch.sum(O * dO,dim=-1)
        S = torch.matmul(Q,K.transpose(-2,-1)) * scale
        if ctx.is_causal:
            mask = torch.tril(torch.ones_like(S,dtype=bool))
            S = torch.where(mask,S,float("-inf"))
        P = torch.exp(S - L[...,None])
        dV = torch.matmul(P.transpose(-2,-1),dO)
        dP = torch.matmul(dO,V.transpose(-2,-1))
        dS = P * (dP - D[...,None])
        dQ = torch.matmul(dS,K) * scale
        dK = torch.matmul(dS.transpose(-2,-1),Q) * scale

        return dQ.view(ctx.q_shape), dK.view(ctx.k_shape), dV.view(ctx.v_shape), None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_lb, stride_ls,
    SEQ_Q, SEQ_K,
    scale, # 1 / sqrt(d)
    D: tl.constexpr,
    TILE_SIZE_Q: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    is_causal: tl.constexpr
):
    q_tile_idx = tl.program_id(1)
    batch_idx = tl.program_id(0)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(SEQ_Q,D),
        strides=(stride_qs,stride_qd),
        offsets=(q_tile_idx * TILE_SIZE_Q,0),
        block_shape=(TILE_SIZE_Q,D),
        order=(1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(SEQ_K,D),
        strides=(stride_ks,stride_kd),
        offsets=(0,0),
        block_shape=(TILE_SIZE_K,D),
        order=(1,0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(SEQ_K,D),
        strides=(stride_vs,stride_vd),
        offsets=(0,0),
        block_shape=(TILE_SIZE_K,D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(SEQ_Q,D),
        strides=(stride_os,stride_od),
        offsets=(q_tile_idx * TILE_SIZE_Q,0),
        block_shape=(TILE_SIZE_Q,D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(SEQ_Q,),
        strides=(stride_ls,),
        offsets=(q_tile_idx * TILE_SIZE_Q,),
        block_shape=(TILE_SIZE_Q,),
        order=(0,)
    )

    Qi = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    mi = tl.full((TILE_SIZE_Q,),value=float("-inf"),dtype=tl.float32)
    li = tl.zeros((TILE_SIZE_Q,),dtype=tl.float32)
    Oi = tl.zeros((TILE_SIZE_Q,D),dtype=tl.float32)

    qi_idx = q_tile_idx * TILE_SIZE_Q + tl.arange(0,TILE_SIZE_Q)
    
    for k_iter in range(tl.cdiv(SEQ_K,TILE_SIZE_K)):
        Kj = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        Vj = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")

        kj_idx = k_iter * TILE_SIZE_K + tl.arange(0,TILE_SIZE_K)
        mask = kj_idx[None,:] < SEQ_K
        if is_causal:
            mask &= (qi_idx[:,None] >= kj_idx[None,:] )

        Sj = tl.dot(Qi,Kj.T) * scale #(seq_q,seq_k)
        Sj = tl.where(mask,Sj,-1e6)
        mj_1 = mi
        mi = tl.maximum(mi,tl.max(Sj,axis=1))
        Pj = tl.exp(Sj - mi[:,None]) 
        weight = tl.exp(mj_1 - mi)
        li = li * weight + tl.sum(Pj,axis=1)
        Pj = Pj.to(dtype=Vj.dtype)
        Oi = Oi * weight[:,None] + tl.dot(Pj,Vj)

        K_block_ptr = K_block_ptr.advance((TILE_SIZE_K,0))
        V_block_ptr = V_block_ptr.advance((TILE_SIZE_K,0))

    Oi = Oi * (1.0 / li[:,None])
    li = mi + tl.log(li)
    tl.store(O_block_ptr,Oi.to(dtype=O_block_ptr.type.element_ty),boundary_check=(0,))
    tl.store(L_block_ptr,li,boundary_check=(0,))

def _flash_backward(Q, K, V, O, dO, L, is_causal: bool):
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    D = torch.sum(O.to(dtype=torch.float32) * dO.to(dtype=torch.float32),dim=-1)
    S = torch.bmm(Q,K.transpose(-2,-1),out_dtype=torch.float32) * scale
    if is_causal:
        mask = torch.tril(torch.ones_like(S,dtype=bool))
        S = torch.where(mask,S,float("-inf"))
    P = torch.exp(S - L[...,None])
    dV = torch.matmul(P.to(dtype=dO.dtype).transpose(-2,-1),dO)
    dP = torch.bmm(dO,V.transpose(-2,-1),out_dtype=torch.float32)
    dS = P * (dP - D[...,None])
    dS = dS.to(dtype=Q.dtype)
    dQ = torch.matmul(dS,K) * scale
    dK = torch.matmul(dS.transpose(-2,-1),Q) * scale

    return dQ,dK,dV

_flash_backward_compiled = torch.compile(_flash_backward) 


class flash_attention_triton_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal=False):
        SEQ_Q = Q.shape[-2]
        SEQ_K = K.shape[-2]
        D = Q.shape[-1]
        
        q_shape = Q.shape
        k_shape = K.shape
        v_shape = V.shape

        Q = rearrange(Q, "... seq_q d -> (...) seq_q d")
        K = rearrange(K, "... seq_k d -> (...) seq_k d")
        V = rearrange(V, "... seq_v d -> (...) seq_v d")

        ctx.q_shape = q_shape
        ctx.k_shape = k_shape
        ctx.v_shape = v_shape
        ctx.TILE_SIZE_Q = 16
        ctx.TILE_SIZE_K = 16
        ctx.is_causal = is_causal
        ctx.bsz = Q.shape[0]

        O = torch.zeros_like(Q)
        L = torch.zeros((ctx.bsz, SEQ_Q),device=Q.device,dtype=torch.float32)

        flash_fwd_kernel[(ctx.bsz,triton.cdiv(SEQ_Q,ctx.TILE_SIZE_Q))](
            Q,K,V,
            O,L,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            O.stride(0),O.stride(1),O.stride(2),
            L.stride(0),L.stride(1),
            SEQ_Q=SEQ_Q,SEQ_K=SEQ_K,
            scale=1.0 / (D ** 0.5),
            D=D,
            TILE_SIZE_Q=ctx.TILE_SIZE_Q,TILE_SIZE_K=ctx.TILE_SIZE_K,
            is_causal=is_causal
        )

        ctx.save_for_backward(Q,K,V,O,L)
        return O.view(q_shape)

    @staticmethod
    def backward(ctx, dO):
        dO = rearrange(dO,"... seq_q d -> (...) seq_q d")
        Q,K,V,O,L = ctx.saved_tensors
        dQ,dK,dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ.view(ctx.q_shape), dK.view(ctx.k_shape), dV.view(ctx.v_shape), None

    

