# ReBeL high-quality training checkpoint

Warm-start with:
```
python scripts/train_scale.py --resume checkpoints/rebel_hq.pt \
  --num-worlds 24 --cfr-iters 60 --depth-limit 6 --full-depth-cards 2 \
  --generations 300 --hands-per-gen 8 --train-steps 25 --eval-every 3 --eval-hands 200
```
Config: 24 worlds, depth 6, exact full-depth for the last 2 tricks, belief on.
Committed periodically for durability against container restarts.
