#!/usr/bin/env python
import yaml

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

lr = cfg['elsa']['learning_rate']
print(f"learning_rate: {lr}, type: {type(lr).__name__}")
print(f"Full ELSA config: {cfg['elsa']}")
