import numpy as np
from fractions import Fraction

def factor(n):
    f = 1
    for x in range(1, n+1):
        f=f*x
    return f

def element(n):
    # return ((-1)**(n-1))/(6**n)
    # return 8/(n**(1/3))
    # return ((-1)**(n-1))/factor(n)
    # return 15/((-4)**n)
    # return np.cos(6*n)
    return n/(n**2 + 1)

def element_sum(n):
    return sum([element(x) for x in range(1, n+1)])

for n in range(1, 11):
    print(element_sum(n))
    # print(Fraction(element_sum(n)).limit_denominator())
