#!/usr/bin/env python
import sys
sys.path.insert(0, "../")

from optimizers.HillClimbing import HillClimbing
from additionalFuncs.evaluation import meanSpeedAdaSOTL

if __name__ == '__main__':
    alpha = 3
    beta = 1.4
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.001, maxIter=2, numRuns=2, strategy=1)