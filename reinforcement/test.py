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
for vec1, vec2 in zip(
        [(-6, 4, 6), (6,2),  (-1, 2, 4), (4, 8, -8)],
        [(6 ,-7, 2), (-1,3), (2, 3, -1), (-6, -12, 12)]):
    # print(mod(vec1))
    # print(mod(vec2))
    print("orthogonal", isOrth(vec1, vec2))
    print("parallel", isParallel(vec1, vec2))
    print()
import numpy as np
print(np.arccos(9/(mod((2*np.sqrt(6), 1))*mod((0,9)))))
b = (3,1,2)
a = (0,2,1)
print(np.cross(a, b))
print(np.dot(crossProd(a, b), a) )
print(np.dot(crossProd(a, b), b) )

a = (2*np.sqrt(6), 1)
b = (0, 9)
print(np.arccos(np.dot(a, b)/(mod(a)*mod(b))) * 180 / np.pi)