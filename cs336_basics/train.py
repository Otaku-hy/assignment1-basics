import torch
import argparse
import json
import os
import math
import time
import wandb

import numpy as np

from cs336_basics.training_infra import *
from cs336_basics.transformer import Transformer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    data = parser.add_argument_group("data")
    data.add_argument("--train-path",type=str, required=True)
    data.add_argument("--val-path",type=str, required=True)

    model = parser.add_argument_group("model")
    model.add_argument("--d-model",type=int, default=512)
    model.add_argument("--num-layers",type=int, default=4)
    model.add_argument("--num-heads",type=int, default=16)
    model.add_argument("--d-ff",type=int, default=1344)
    model.add_argument("--num-q-per-group",type=int, required=None)
    model.add_argument("--rope-theta",type=float, default=10000.0)
    model.add_argument("--vocab-size",type=int, default=10000)
    model.add_argument("--context-length",type=int, default=256)

    optim = parser.add_argument_group("optimizer")
    optim.add_argument("--max-lr",type=float,required=True)
    optim.add_argument("--min-lr",type=float,default=None)
    optim.add_argument("--warmup-iters",type=int,default=0)
    optim.add_argument("--cosine-iters",type=int,default=None)
    optim.add_argument("--betas", type=float, nargs=2, default=(0.9,0.95))
    optim.add_argument("--weight-decay",type=float,default=0.1)
    optim.add_argument("--eps",type=float,default=1e-8)
    optim.add_argument("--grad-clip", type=float, default=1.0)

    train = parser.add_argument_group("training")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--max-iters",type=int, required=True)
    train.add_argument("--device",type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--grad-ckpt",action="store_true")

    log = parser.add_argument_group("logging")
    log.add_argument("--log-interval", type=int, default=100)
    log.add_argument("--eval-interval", type=int, default=500)
    log.add_argument("--eval-iters", type=int, default=50)
    log.add_argument("--wandb-project", type=str, default=None)
    log.add_argument("--run-name", type=str, default=None)


    ckpt = parser.add_argument_group("checkpoint")
    ckpt.add_argument("--out-dir", type=str, required=True)
    ckpt.add_argument("--save-interval", type=int, default=1000)
    ckpt.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    if args.min_lr is None:
        args.min_lr = args.max_lr / 10
    if args.cosine_iters is None:
        args.cosine_iters = args.max_iters
    return args

class RunLogger:
    def __init__(self, out_dir: str, config: dict, wandb_project: str | None = None, run_name: str | None = None):
        self.path = os.path.join(out_dir, "log.jsonl")
        self.start = time.perf_counter()
        self.wandb = None
        if wandb_project is not None:
            wandb.init(project=wandb_project, name=run_name, config=config, dir=out_dir)
            self.wandb = wandb

    def log(self, **record) -> None:
        record["time"] = time.perf_counter() - self.start
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb is not None:
            self.wandb.log(record, step=record["step"])

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def evaluate(model:torch.nn.Module,batches:list[torch.Tensor,torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        loss = 0
        for x, targets in batches:
            loss += cross_entropy(model(x),targets)
        loss /= len(batches)
    model.train()
    return loss.item()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = np.memmap(args.train_path,dtype=np.uint16,mode="r")
    valid_data = np.memmap(args.val_path,dtype=np.uint16,mode="r")

    model = Transformer(args.d_model,args.num_heads,args.d_ff,args.rope_theta,args.vocab_size,args.context_length,args.num_layers,args.num_q_per_group,args.device,torch.float32,args.grad_ckpt)
    model.compile()
    
    optim = AdamW(model.parameters(),lr=args.max_lr,betas=args.betas,weight_decay=args.weight_decay,eps=args.eps)

    start_it = 0
    if args.resume is not None:
        start_it = load_checkpoint(args.resume, model, optim)

    eval_batches = [load_data(valid_data, args.batch_size,args.context_length,args.device) for i in range(args.eval_iters)]

    logger = RunLogger(args.out_dir, vars(args), args.wandb_project, args.run_name)

    running_loss, steps_since_log = 0.0, 0
    t_last = time.perf_counter()

    # test overfit
    #x, targets = load_data(train_data,args.batch_size,args.context_length,args.device)

    #training loop
    for it in range(start_it, args.max_iters):
        model.train()

        lr = lr_cosine_schedule(it,args.max_lr,args.min_lr,args.warmup_iters,args.cosine_iters)
        for group in optim.param_groups: group["lr"] = lr 

        x, targets = load_data(train_data,args.batch_size,args.context_length,args.device)
        
        loss = cross_entropy(model(x),targets)

        optim.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(),args.grad_clip)
        optim.step()

        # logging
        running_loss += loss.detach()
        steps_since_log += 1
        if (it + 1) % args.log_interval == 0 or it == start_it:
            avg_loss = (running_loss / steps_since_log).item()
            now = time.perf_counter()
            tokens = steps_since_log * args.batch_size * args.context_length
            logger.log(step=it + 1, train_loss=avg_loss, lr=lr, tokens_per_s=tokens / (now - t_last))
            if not math.isfinite(avg_loss):
                raise RuntimeError(f"loss became {avg_loss} at step {it + 1}")
            running_loss, steps_since_log, t_last = 0.0, 0, now


        # eval
        if (it +1) % args.eval_interval == 0 or it + 1 == args.max_iters:
            val_loss = evaluate(model,eval_batches)
            logger.log(step=it +1, val_loss = val_loss, val_ppl = math.exp(val_loss))
            t_last = time.perf_counter()

        # ckpt
        if (it +1) % args.save_interval == 0 or it + 1 == args.max_iters:
            save_checkpoint(model,optim,it+1,os.path.join(args.out_dir, f"ckpt_{it+1}.pt"))

    logger.finish()