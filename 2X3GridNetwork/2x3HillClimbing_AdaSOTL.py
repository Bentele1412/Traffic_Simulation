#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    alpha = 4.182
    beta = 1.1866
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.1, 0.025]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.01, maxIter=50, numRuns=5, strategy=1)