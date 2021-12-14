#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    ctFactor = 0.9
    phaseShifts = [10, 20, 10, 20, 30]
    evalFunc = meanSpeedCycleBased
    
    params = [ctFactor] + phaseShifts
    stepSizes = [0.1] + [5]*5
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.1, maxIter=50, numRuns=10)