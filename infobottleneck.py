# Create an N x N magic square. N must be odd.
import sys

import numpy as np


def generate_discreet_prob(k):
    arr = np.random.rand(k)
    # arr += abs(np.min(arr))
    return  arr / np.sum(arr)


def add_dependence(pAB, magnitude=0.1, step=0.1):
    assert step < 1 and step > 0
    cum_diff = 0
    det_events = 0
    while cum_diff < magnitude and det_events < len(pAB):
        # pick an event B where to add the dependence
        ai = np.random.randint(0, len(pAB))
        # from that event, pick a dependent event from A
        b_increase_i = np.random.randint(0, len(pAB))
        # get the total probability of the event
        b_prob = np.sum(pAB[ai])
        # increase the probability of the dependent event by a percentage
        diff = step * b_prob
        # make sure the update step size is not too big for the specific event
        diff = min(diff, b_prob - pAB[ai][b_increase_i])
        # if can't increase the probability, skip and try another event
        if diff == 0:
            det_events += 1
            continue
        pAB[ai][b_increase_i] += diff
        ndiff = diff / len(pAB-1)
        over_diff = 0
        for j in range(len(pAB)):
            if j != b_increase_i:
                if pAB[ai][j] < (ndiff + over_diff):
                    over_diff = (ndiff + over_diff) - pAB[ai][j]  # add the difference to the next event
                    pAB[ai][j] = 0
                else:
                    pAB[ai][j] -= (ndiff + over_diff)
                    over_diff = 0
        cum_diff += diff

    return pAB


def I(pA, pB, pAB):
    """mutual information, or DKL(P(X,Y) || PX x PY)
    :param pA: A - p(a)
    :param pB: B - p(b)
    :param pAB: A - p(a,b)"""
    r = 0
    for ia, pa in enumerate(pA):
        for ib, pb in enumerate(pB):
            r += pAB[ia][ib] * np.log( max(1e-24, pAB[ia][ib]) / (max(1e-24, pa * pb)) )
    return r


def dkl_yx_yt(pY_X, pY_T):
    """
    Kulback-Leibler divergence between mappings from input to target distribution
    and from latent codes to target distribution
    :param pYX: p(y|x), input to target distribution
    :param pYT: p(y|t), latent codes to target distribution
    :return: DKL(p(y|x) || p(y|t))
    """
    return np.sum(pY_X * np.log(max(1e-24, pY_X / pY_T)))


def entropy(p):
    """entropy of a discrete distribution"""
    return -np.sum(p * np.log(p))


def visualize_dependence(pAB, pA, pB):
    """visualize the dependence of the events in the probability matrix"""
    import matplotlib.pyplot as plt
    dep_mat = np.zeros(pAB.shape)
    for papb in [(pA, pB), (pB, pA)]:
        # events A
        for ia, pa in enumerate(papb[0]):
            for ib, pb in enumerate(papb[1]):
                dep_mat[ia][ib] = pAB[ia][ib] / papb[1][ib] - papb[0][ia]
        plt.figure(figsize=(10, 10))
        plt.imshow(pAB, cmap="CMRmap", interpolation='nearest', vmin=0, vmax=pAB.max())

        plt.show()


def probs():

    A = generate_discreet_prob(50)
    B = generate_discreet_prob(50)
    # independent case
    AB = np.outer(A, B)
    print(AB)
    print("MI of independent distributions", I(A, B, AB))

    # dependent case
    AB = np.outer(A, B)
    # introduce dependence
    AB = add_dependence(AB, 0.5)
    print(AB)
    print("MI of dependent distributions", I(A, B, AB))
    visualize_dependence(AB, A, B)
    # define encoder

    AT = np.random.rand()


def iterative_information_bottleneck(pXY, beta, M):
    """
    iterative IB from Slonim IB Theory and Applications paper
    see Figure 3.1 for pseudo code
    we are running the algorithm a fixed number of iterations instead of checking convergence
    :param pXY: p(x, y)
    :param beta: tradeoff parameter
    :param M: cardinality of the bottleneck
    :return: A typically soft partition T of X into M parts
    """
    assert beta >= 0
    assert M >= 0

    # compute p(y | x)
    pY_X = np.sum(pXY, axis=1)

    # initialize random p(t | x)
    pT_X = np.random.rand(len(pXY), M)

    # compute pT according to equation 3.2
    # P_m(t) = sum_x ( p(x) P_m(t | x) )
    pT = np.zeros((M,))
    for i in range(M):
        pT[i] = np.sum( np.sum(pXY[ix,:]) * pT_X[ix][i] for ix, px in enumerate(pXY))

    # compute p(y | t)
    # P_m(y | t) = sum_x ( P_m(t | x) p(x,y) ) / P_m(t)
    pY_T = np.random.rand(M, len(pXY))
    for i in range(M):
        pY_T[i] = np.sum(pT_X[ix][i] * pXY[ix] for ix, px in enumerate(pXY)) / pT[i]

    sum_px = [np.sum(pXY[ix]) for ix in range(len(pXY))]
    # loop
    print("iterative IB")
    ITERS = 1000
    ticks = 10
    tick = (ITERS//ticks)
    for _ in range(1, ITERS+1):
        if _ % tick == 0:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            n = _//tick
            sys.stdout.write(f"[{'='*n}{' '*(ticks-n)}] {_}/{ITERS} iterations")
            # sys.stdout.write("[%-10s] %d%%" % ('=' * (_//tick), _))
            sys.stdout.flush()
            # print("iteration", _, end="\r")
        for it in range(M):
            for ix, px in enumerate(sum_px):
                # DKL(P(y|x) || P(y|t))
                dkl = dkl_yx_yt(pY_X[ix], pY_T[it][ix])

                pT_X[ix][it] = pT[it] / ( np.random.normal( px, beta ) * np.exp(beta * dkl) )

            pT[it] = np.sum(pT_X[ix])



if __name__ == '__main__':
    X = generate_discreet_prob(50)
    Y = generate_discreet_prob(50)
    XY = np.outer(X, Y)
    XY = add_dependence(XY, 0.5)

    iterative_information_bottleneck(XY, beta=0.5, M=25)