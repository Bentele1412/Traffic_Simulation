#!/usr/bin/env python

from helperTwoPhases import *

if __name__ == '__main__':
    theta = 30
    evalFunc = meanSpeedSOTL
    
    params = [theta]
    stepSizes = [2]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.01, maxIter=50, numRuns=5, strategy=1)