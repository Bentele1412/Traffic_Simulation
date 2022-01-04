#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    alpha = 5
    beta = 1.5
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [1, 0.2]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.1, maxIter=5, numRuns=1, strategy=1)