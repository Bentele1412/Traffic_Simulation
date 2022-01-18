#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    alpha = 3
    beta = 1.4
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.001, maxIter=100, numRuns=5, strategy=1)