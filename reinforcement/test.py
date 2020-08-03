import torch.nn as nn
import torch.optim as optim
import torch

inp = 4
hidden = 20
out = 5

policy = nn.Sequential(
    nn.Linear(inp, hidden),
    nn.ReLU(),
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out),
    nn.Sigmoid()
).cuda()
device = torch.device("cuda")
print(device)
policy.to(device)


optimizer = optim.Adam(policy.parameters(), lr=0.001)

outs = policy.forward(torch.tensor([0.2, 0.3, 0.4, 0.6], device=device))

