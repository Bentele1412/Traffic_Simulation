#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    ctFactor = 0.9
    #ctFactor = 1.6726
    phaseShifts = [10, 20, 10, 20, 30]
    #phaseShifts = [10, 158.49, 424.63, -813.05, 30]
    evalFunc = meanSpeedCycleBased
    
    params = [ctFactor] + phaseShifts
    stepSizes = [0.1] + [1]*5
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.1, maxIter=20, numRuns=1, strategy=1)