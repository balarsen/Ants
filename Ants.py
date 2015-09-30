
"""
I am rolling an ant algorithm to try and learn more on what this is about

setup:
- make a grid 256x256 that the ants can walk on, they release a phermone that
    makes them more likely to walk that way. There is also a dander phermone
    that makes them less likely.
    

"""
import time


import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import spacepy.toolbox as tb


def setup(shape=(256, 256)):
    board = lil_matrix(shape, dtype=np.int8)
    return board

class Ant(object):
    def __init__(self, loc=(0,0), pher=None, carrying=False):
        self.loc = np.asarray(loc)
        self.pher = pher
        self.carrying = carrying

    def __str__(self):
        return('<Ant> ({0},{1})'.format(self.loc[0], self.loc[1]))

    __repr__ = __str__

    def getNeighbors(self):
        # not an edge
        if ((self.loc[0] not in [0,self.pher.shape[0]-1]) and
            (self.loc[1] not in [0,self.pher.shape[1]-1])):
            edges = np.asarray([(self.loc[0]-1, self.loc[1]-1),
                                (self.loc[0]-1, self.loc[1]),
                                (self.loc[0]-1, self.loc[1]+1),
                                (self.loc[0], self.loc[1]-1),
                                (self.loc[0], self.loc[1]+1),
                                (self.loc[0]+1, self.loc[1]-1),
                                (self.loc[0]+1, self.loc[1]),
                                (self.loc[0]+1, self.loc[1]+1)])
        elif (self.loc[0] == 0):
            if (self.loc[1] not in [0,self.pher.shape[1]-1]):
                edges = np.asarray([(self.loc[0], self.loc[1]-1),
                                    (self.loc[0], self.loc[1]+1),
                                    (self.loc[0]+1, self.loc[1]-1),
                                    (self.loc[0]+1, self.loc[1]),
                                    (self.loc[0]+1, self.loc[1]+1)])
            elif (self.loc[1] == 0):
                edges = np.asarray([(self.loc[0], self.loc[1]+1),
                                    (self.loc[0]+1, self.loc[1]),
                                    (self.loc[0]+1, self.loc[1]+1)])
            elif (self.loc[1] == self.pher.shape[1]-1):
                edges = np.asarray([(self.loc[0], self.loc[1]-1),
                                    (self.loc[0]+1, self.loc[1]-1),
                                    (self.loc[0]+1, self.loc[1])])
        elif (self.loc[0] == self.pher.shape[0]-1):
            if (self.loc[1] not in [0,self.pher.shape[1]-1]):
                edges = np.asarray([(self.loc[0]-1, self.loc[1]-1),
                                    (self.loc[0]-1, self.loc[1]),
                                    (self.loc[0]-1, self.loc[1]+1),
                                    (self.loc[0], self.loc[1]-1),
                                    (self.loc[0], self.loc[1]+1)])
            elif (self.loc[1] == 0):
                edges = np.asarray([(self.loc[0]-1, self.loc[1]),
                                    (self.loc[0]-1, self.loc[1]+1),
                                    (self.loc[0], self.loc[1]+1)])
            elif (self.loc[1] == self.pher.shape[1]-1):
                edges = np.asarray([(self.loc[0]-1, self.loc[1]-1),
                                    (self.loc[0]-1, self.loc[1]),
                                    (self.loc[0], self.loc[1]-1)])
        elif (self.loc[1] == 0):
            edges = np.asarray([(self.loc[0]-1, self.loc[1]),
                                (self.loc[0]-1, self.loc[1]+1),
                                (self.loc[0], self.loc[1]+1),
                                (self.loc[0]+1, self.loc[1]),
                                (self.loc[0]+1, self.loc[1]+1)])
        elif (self.loc[1] == self.pher.shape[1]-1):
            edges = np.asarray([(self.loc[0]-1, self.loc[1]-1),
                                (self.loc[0]-1, self.loc[1]),
                                (self.loc[0], self.loc[1]-1),
                                (self.loc[0]+1, self.loc[1]-1),
                                (self.loc[0]+1, self.loc[1])])            
        else:
            raise(ValueError("Should not have gotten here"))
        return edges

    def getNeighborsWeights(self, edges=None):
        if edges is None:
            edges = self.getNeighbors()
        weights = [self.pher[i1, i2] for i1, i2 in edges]
        return np.asarray(weights)

    def normalizeWeights(self, weights):
        # make the weights sume to one
        if not weights.any():
            p1 = None
        else:
            # starting probability
            p = len(weights)/weights.sum()
            p1 = np.ones(len(weights))
            # then for the ones with higher weight add some
            ind = np.nonzero(weights)
            p1[ind] *= weights[ind]
            p1 /= p1.sum()
        return p1
            
    def selectMoveNeighbor(self):
        # first need the neighbors
        neighbors = self.getNeighbors()
        # and the neighbor weights
        weights = self.getNeighborsWeights(edges=neighbors)
        nwei = self.normalizeWeights(weights)
        if nwei is None:
            mv = choice(np.arange(len(neighbors)), 1)
        else:
            mv = choice(np.arange(len(neighbors)), 1, nwei.tolist())
        return neighbors[mv]

    def move(self, food):
        if self.loc == food:
            self.
        self.loc = self.selectMoveNeighbor().reshape(2)



def plotAnts(pher, ants, food=None, hole=None, fig=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = fig.gca()
        ax.clear()
    ax.pcolormesh(pher.toarray())
    dat = []
    for a in ants:
        dat.append(a.loc)
    dat = np.asarray(dat).reshape(-1, 2)
    ax.scatter(dat[:,0], dat[:,1], c='y', linewidths=0)
    if food is not None:
        ax.scatter(food[0], food[1], c='g', s=50, linewidths=0)
    if hole is not None:
        ax.scatter(hole[0], hole[1], c='r', s=50, linewidths=0)
    plt.draw()
    return fig
    
    
    
            
if __name__ == '__main__':
    pher = setup((50,50))
    food = np.asarray([40,40])
    hole = (20,20)
    ants = []
    for i in range(50):        # make an Ant at the hole
        ants.append(Ant(loc=hole, pher=pher))

    fig = plotAnts(pher, ants)
    while True:
        # move the ant
        for a in ants:
            a.move()
        fig = plotAnts(ants, fig=fig, food=food, hole=hole)
        time.sleep(0.25)




