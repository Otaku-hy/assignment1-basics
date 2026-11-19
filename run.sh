#!/usr/bin/env bash
# 粗扫描 LR，用法: bash sweep_lr.sh
LRS="1e-4 3e-4 1e-3 3e-3 1e-2"
OUT_ROOT=runs/owt/lr_coarse
COMMON="--train-path data/owt_train_token.bin --val-path data/owt_valid_token.bin --vocab-size 32000 --batch-size 64 --log-interval 40 --eval-interval 200 --eval-iters 40 --save-interval 10000 --max-iters 4000 --warmup-iters 200"

for lr in $LRS; do
  out="$OUT_ROOT/lr_$lr"
  mkdir -p "$out"
  echo "=== lr=$lr ==="
  uv run python -m cs336_basics.train $COMMON --max-lr $lr --out-dir $out --wandb-project cs336-a1 --run-name owt_lr_$lr \
    > "$out/stdout.log" 2>&1 || echo "lr=$lr 报错退出（可能发散），看 $out/stdout.log"
  grep val_loss "$out/log.jsonl" | tail -n 1
done
