'''
Forward-backward algorithm for entity linking problem

Top K entity candidates are pre-identified for each mention.
One entity candidate is specified as NULL. Mentions can overlap
with each other, we need to handle mention overlapping if
both mentions are linked to real entities.

Author: Yi Yang
Email: yangyiycc@gmail.com
'''

import numpy as np
from copy import copy

def forward(potentials, beginEndIndices):
    '''
    potentials is a list of lists, and each entry contains the potential 
    function value for a entity candidate of a mention.
    beginEndIndices is a list of tuples, and each tuple contains the
    begin index and end index of a mention.
    '''
    
    orders = []
    lnoms = []
    sortedIndices = sorted(beginEndIndices, key=lambda indices:indices[2]) # sorting according to end index
    #print(sortedIndices)
    Alpha = []
    for i in xrange(0, len(potentials)):
        orders.append(sortedIndices[i][0])
        alpha = []
        if i == 0: 
            alpha.append(np.exp(potentials[sortedIndices[i][0]][0]))
        else:
            alpha.append(np.exp(potentials[sortedIndices[i][0]][0]) * sum(Alpha[i-1]))
        lnom = i - 1 # last nonoverlap mention
        while lnom >= 0 and sortedIndices[i][1] < sortedIndices[lnom][2]:
            lnom = lnom - 1
        lnoms.append(lnom)
        for j in xrange(1, len(potentials[sortedIndices[i][0]])): # all non NULL entities
            prodVal = 1
            if lnom >= 0:
                prodVal = prodVal * sum(Alpha[lnom]) # sum over alphas of last nonoverlap mention
            ii = i - 1
            while ii > lnom: # potential values of NULL entities of overlap mentions
                prodVal = prodVal * np.exp(potentials[sortedIndices[ii][0]][0])
                ii = ii - 1
            prodVal = prodVal * np.exp(potentials[sortedIndices[i][0]][j])
            alpha.append(prodVal)
        #print(alpha)
        Alpha.append(alpha)
        
    return (Alpha, orders, lnoms)


def backward(potentials, beginEndIndices):
    '''
    potentials is a list of lists, and each entry contains the potential 
    function value for a entity candidate of a mention.
    beginEndIndices is a list of tuples, and each tuple contains the
    begin index and end index of a mention.
    '''
    
    orders = []
    nnoms = []
    sortedIndices = sorted(beginEndIndices, key=lambda indices:indices[1], reverse=True) # sorting according to begin index
    #print(sortedIndices)
    Beta = []
    for i in xrange(0, len(potentials)):
        orders.append(sortedIndices[i][0])
        beta = []
        if i == 0: 
            beta.append(1)
        else:
            sumVal = 0
            for j in xrange(0, len(potentials[sortedIndices[i-1][0]])): 
                sumVal = sumVal + np.exp(potentials[sortedIndices[i-1][0]][j]) * Beta[i-1][j]
            beta.append(sumVal)
        nnom = i - 1 # next nonoverlap mention
        while nnom >= 0 and sortedIndices[i][2] > sortedIndices[nnom][1]:
            nnom = nnom - 1
        nnoms.append(nnom)
        for j in xrange(1, len(potentials[sortedIndices[i][0]])): # all non NULL entities
            prodVal = 1
            if nnom >= 0:
                prodVal = 0
                for k in xrange(0, len(potentials[sortedIndices[nnom][0]])): 
                    prodVal = prodVal + np.exp(potentials[sortedIndices[nnom][0]][k]) * Beta[nnom][k] 
            ii = i - 1
            while ii > nnom: # potential values of NULL entities of overlap mentions
                prodVal = prodVal * np.exp(potentials[sortedIndices[ii][0]][0])
                ii = ii - 1
            beta.append(prodVal)
        #print(beta)
        Beta.append(beta)
        
    return (Beta, orders, nnoms)
    


def compute_probs(potentials, beginEndIndices):
    (Alpha, forwardOrders, lnoms) = forward(potentials, beginEndIndices)
    (Beta, backwardOrders, nnoms) = backward(potentials, beginEndIndices)
    Z = sum(Alpha[len(Alpha)-1])
    print(Z)
    Probs = []
    for i in xrange(0, len(potentials)): Probs.append([])
    for i in xrange(0, len(forwardOrders)):
        idx = forwardOrders[i]
        overlapNulValue = 1
        for j in xrange(0, len(beginEndIndices)):
            if j == idx: continue
            if not(beginEndIndices[j][1] >= beginEndIndices[idx][2] or beginEndIndices[idx][1] >= beginEndIndices[j][2]): # overlap
                overlapNulValue = overlapNulValue * np.exp(potentials[j][0])
        localZ = overlapNulValue 
        if lnoms[i] >= 0: 
            localZ = localZ * sum(Alpha[lnoms[i]])
        corIdx = -1
        for j in xrange(0, len(backwardOrders)): # find corresponding index in backwardOrders for forwardOrders[i]
            if backwardOrders[j] == forwardOrders[i]:
                corIdx = j
        if nnoms[corIdx] >= 0:
            sumVal = 0
            for j in xrange(0, len(Beta[nnoms[corIdx]])): 
                sumVal = sumVal + np.exp(potentials[backwardOrders[nnoms[corIdx]]][j]) * Beta[nnoms[corIdx]][j]
            localZ = localZ * sumVal
        Probs[idx].append(0)
        localZs = [0]
        sumZ = 0
        for j in xrange(1, len(potentials[idx])):
            sumZ = sumZ + localZ * np.exp(potentials[idx][j])
            Probs[idx].append(localZ * np.exp(potentials[idx][j]) / Z)
        Probs[idx][0] = 1 - sumZ/Z
    print(Probs)

    return Probs

def violence_search(potentials, beginEndIndices):
    sortedIndices = sorted(beginEndIndices, key=lambda indices:indices[2]) # sorting according to end index
    paths = [[]]
    lIdx = [-1]
    gains = [1]
    for i in xrange(0, len(potentials)):
        idx = sortedIndices[i][0]
        newPaths = []
        newLIdx = []
        newGains = []
        for j in xrange(0, len(paths)):
            #print(j)
            #print(paths)
            #print(paths[j])
            thispath = copy(paths[j])
            thispath.append(0)
            newPaths.append(thispath)
            #print(newPaths)
            newLIdx.append(lIdx[j])
            newGains.append(gains[j]*np.exp(potentials[idx][0]))
            for k in xrange(1, len(potentials[idx])):
                #print(k)
                #print(beginEndIndices[idx][1])
                #print(lIdx[j])
                if beginEndIndices[idx][1] < lIdx[j]: continue
                thispath = copy(paths[j])
                thispath.append(k)
                #print(thispath)
                newPaths.append(thispath)
                newLIdx.append(beginEndIndices[idx][2])
                newGains.append(gains[j]*np.exp(potentials[idx][k]))
        paths = newPaths[:]
        lIdx = newLIdx[:]
        gains = newGains[:]
        
    Probs = []
    Z = sum(gains)
    print(Z)
    for i in xrange(0, len(potentials)): Probs.append([])
    for i in xrange(0, len(potentials)):
        idx = sortedIndices[i][0]
        probs = []
        for j in xrange(0, len(potentials[idx])): probs.append(0)
        for j in xrange(0, len(paths)):
            probs[paths[j][i]] = probs[paths[j][i]] + gains[j]/Z
        Probs[idx] = probs

    print(Probs)


# potentials
potentials = [[1,2,1], [2,1,3,1], [1,1,5], [1,1,3]]
#potentials = [[1,1], [1,1], [1,1], [1,1,3]]
# begin and end index
#indices = [(0,1,5), (1,1,3), (2,2,3), (3,3,4)]
indices = [(0,2,5), (1,1,3), (2,2,3), (3,3,4)]

if __name__ == '__main__':
    compute_probs(potentials, indices)
    violence_search(potentials, indices)
