import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

cons = nn.Linear(10, 20)

prune.random_unstructured(cons, name="weight",  amount=0.1)
print(cons.weight)
# mask.scatter_(2, cons, 1.)

# print(mask)
