#!/usr/bin/env python

import sys
sys.path.insert(0, "../")

from optimizers.HillClimbing import HillClimbing
from additionalFuncs.evaluation import meanSpeedSOTL

if __name__ == '__main__':
    theta = 30
    evalFunc = meanSpeedSOTL
    
    params = [theta]
    stepSizes = [1]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.01, maxIter=3, numRuns=1, strategy=1)