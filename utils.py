import numpy as np
import random


def compAcc(tpy, ty):
    return (np.argmax(tpy, axis=1) == np.argmax(ty, axis=1)).sum() \
            * 1.0 / tpy.shape[0]


def getMaxAcc(max_test_acc, acc):
    return max_test_acc if acc < max_test_acc else acc


def subPath(ind1, ind2, path, ws):
    s = []
    m = []
    dif = abs(ind1 - ind2)
    if ind1 > ind2:
        for i in range(ind2, ind1 + 1)[::-1]:
            s.append(int(path[i]))
            m.append(1)
        if dif < ws:
            for j in range(ws-dif):
                s.append(0)
                m.append(0)
        return s, m
    if ind1 < ind2:
        for i in range(ind1, ind2+1):
            s.append(int(path[i]))
            m.append(1)
        if dif < ws:
            for j in range(ws-dif):
                s.append(0)
                m.append(0)
        return s, m
    if ind1 == ind2:
        for i in range(ws+1):
            s.append(0)
            m.append(0)
        s[0] = int(path[ind1])
        m[0] = 1
        return s, m


def samplePathInd(k, path_length, graph, num_ver):
    path = [k]
    for _ in range(path_length):
        le = len(graph[path[-1]])
        for i in range(le):
            nextNode = random.choice(graph[path[-1]])
            if nextNode < num_ver:
                break
            elif i == le - 1:
                nextNode = path[-1]
                break
        path.append(nextNode)
    return path
