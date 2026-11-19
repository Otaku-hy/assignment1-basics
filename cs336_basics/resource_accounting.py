import torch

def resource_account(d_model:int, 
                     num_heads:int, 
                     d_ff:int, vocab_size:int, 
                     context_length:int, 
                     num_layers:int, 
                     num_q_per_group:int=None,
                     dtype = torch.float32,
                     batch_size:int = 1
                     ):
    d_k = d_v = d_model // num_heads

    #### params

    #input embedding
    embedding_params = vocab_size * d_model
    
    #each block
    ln1_params = d_model
    mha_params = d_model * num_heads * d_k * 2 + d_model * num_heads * d_v + num_heads * d_v * d_model #Wq, Wk, Wv, Wo 
    if num_q_per_group is not None:
        mha_params = d_model * num_heads * d_k + d_model * num_heads // num_q_per_group * d_v + d_model * num_heads//num_q_per_group * d_v + num_heads * d_v * d_model #Wq, Wk, Wv, Wo 
    ln2_params = d_model
    ff_params = d_model * d_ff * 2 + d_ff * d_model #Wg,Wv,Wo
    transformer_block_params = ln1_params + mha_params + ln2_params + ff_params
    
    #final ln
    final_ln_params = d_model
    
    #final lm
    final_lm_params = d_model * vocab_size
    
    #not learnable rope table:
    rope_table = context_length * d_model * 2 #cos and sin table

    total_learnable = embedding_params + transformer_block_params * num_layers + final_ln_params + final_lm_params

    #### flops
    embedding_flops = 0

    ln1_flops = 0 #now only count for matmul

    proj_flops = (context_length * d_model * num_heads * d_k * 2) + (context_length * d_model * num_heads * d_k * 2) + context_length * d_model * num_heads * d_v * 2 + context_length * num_heads * d_v * d_model * 2
    if num_q_per_group is not None:
        proj_flops = (context_length * d_model * num_heads * d_k * 2) + (context_length * d_model * num_heads // num_q_per_group * d_k * 2) + context_length * d_model * num_heads//num_q_per_group * d_v * 2 + context_length * num_heads * d_v * d_model * 2
    attn_flops = context_length * d_k * context_length * 2 + context_length * context_length * d_v * 2
    multihead_attn_flops = num_heads * attn_flops
    mha_flops = proj_flops + multihead_attn_flops

    ln2_flops = 0

    ff_flops = context_length * d_model * d_ff * 2 * 3

    transformer_block_flops = ln1_flops + mha_flops + ln2_flops + ff_flops

    final_ln_flops = 0

    final_lm_flops = context_length * d_model * vocab_size * 2

    total_flops = embedding_flops + transformer_block_flops * num_layers + final_ln_flops + final_lm_flops
    
    ####memory_peak - assume AdamW with fp32 used
    bytes_per_param = dtype.itemsize

    params_mem = total_learnable * bytes_per_param
    grad_mem = params_mem
    optimizer_state_mem = total_learnable * 4 * 2 #fp32 for m & v

    ln1_activation = batch_size * context_length * d_model * bytes_per_param
    proj_activation = batch_size * (
        context_length * num_heads * d_k * 2 * bytes_per_param
        + context_length * num_heads * d_v * bytes_per_param
        + context_length * num_heads * d_k * bytes_per_param
    )
    if num_q_per_group is not None:
        proj_activation = batch_size * (
            context_length * num_heads * d_k * bytes_per_param
            + context_length * num_heads * d_k // num_q_per_group * bytes_per_param
            + context_length * num_heads * d_v // num_q_per_group * bytes_per_param
            + context_length * num_heads * d_k * bytes_per_param
        )
    attn_activation = batch_size * num_heads * (
        context_length * context_length * 2 * bytes_per_param
        + context_length * d_v * bytes_per_param)
    mha_activation = proj_activation + attn_activation
    ln2_activation = batch_size * context_length * d_model * bytes_per_param
    ff_activation = batch_size * (
        context_length * d_ff * 2 * bytes_per_param
        + context_length * d_ff * 2 * bytes_per_param
        + context_length * d_model * bytes_per_param
    )
    transformer_block_activation = (ln1_activation + mha_activation + ln2_activation + ff_activation) * num_layers

    final_ln_activation = batch_size * context_length * d_model * bytes_per_param
    final_lm_activation = batch_size * context_length * vocab_size * bytes_per_param
    cross_entropy_activation = batch_size * context_length * vocab_size * bytes_per_param

    total_activation = transformer_block_activation + final_ln_activation + final_lm_activation + cross_entropy_activation

    total_mem = params_mem + grad_mem + optimizer_state_mem + total_activation

    params_parts = {
        "proj (q/k/v/o)": mha_params * num_layers,
        "multihead attn": 0,
        "ff": ff_params * num_layers,
        "final lm": final_lm_params,
    }
    flops_parts = {
        "proj (q/k/v/o)": proj_flops * num_layers,
        "multihead attn": multihead_attn_flops * num_layers,
        "ff": ff_flops * num_layers,
        "final lm": final_lm_flops,
    }
    static_bytes = 2 * bytes_per_param + 4 * 2  # param + grad + fp32 m & v
    mem_parts = {
        "proj (w+g+opt)": mha_params * num_layers * static_bytes,
        "attn (w+g+opt)": 0,
        "ff (w+g+opt)": ff_params * num_layers * static_bytes,
        "lm (w+g+opt)": final_lm_params * static_bytes,
        "proj (act)": proj_activation * num_layers,
        "attn (act)": attn_activation * num_layers,
        "ff (act)": ff_activation * num_layers,
        "lm (act)": final_lm_activation,
    }


    return {
        "params": {"total": total_learnable, "parts": params_parts},
        "flops": {"total": total_flops, "parts": flops_parts},
        "mem": {"total": total_mem, "parts": mem_parts},
    }



FLOP_UNITS = [("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6)]
PARAM_UNITS = [("B", 1e9), ("M", 1e6), ("K", 1e3)]
MEM_UNITS = [("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]


def human(n: float, units: list[tuple[str, float]]) -> str:
    for suffix, scale in units:
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:,.0f}"


def report(title: str, section: dict, units: list[tuple[str, float]]) -> None:
    total, parts = section["total"], section["parts"]
    print(f"== {title}: {human(total, units)} ==")
    rows = dict(parts)
    other = total - sum(parts.values())
    if other:
        rows["other"] = other
    for name, value in rows.items():
        print(f"  {name:<16}{human(value, units):>12}{value / total:>8.1%}")
    if parts:
        print(f"  largest: {max(parts, key=parts.get)}")
    print()


if __name__ == "__main__":
    result = resource_account(
        d_model=1280,
        num_heads=20,
        d_ff=5120,
        vocab_size=10000,
        num_q_per_group=1,
        context_length=4096,
        num_layers=36,
        batch_size=1
    )

    report("params", result["params"], PARAM_UNITS)
    report("FLOPs", result["flops"], FLOP_UNITS)
    report("memory", result["mem"], MEM_UNITS)
