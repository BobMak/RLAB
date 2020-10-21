import math

def mod(a):
    res = 0
    for x in a:
        res += x**2
    return math.sqrt(res)

def isOrth(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i]*b[i]
    if res == 0:
        return True
    else: return False

def isParallel(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    if abs(res) == mod(a) or abs(res) == mod(b):
        return True
    else: return False

def crossProd(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])



import torch
actions = [1, 2, 0]
states  = [0.2, 0.4]
ac = torch.as_tensor(actions,  dtype=torch.float32)
st = torch.as_tensor(states,  dtype=torch.float32)
a  = torch.cat([ac, st], dim=0)
print(a)

