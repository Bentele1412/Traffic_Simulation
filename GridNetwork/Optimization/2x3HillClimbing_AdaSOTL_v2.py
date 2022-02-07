#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedAdaSOTL_v2
from GridNetwork.additionalFuncs.helper import setFlows, checkCTFactor, checkAdaSOTL_v2_params
import random

if __name__ == '__main__':
    random.seed(32)
    '''
    decayRate       = random.uniform(0.1, 1)
    k               = random.uniform(10.0, 50.0)
    alpha           = random.uniform(1.0, 6.0)
    beta            = random.uniform(1.0, 2.0)
    '''
   
    k               = 43.0
    alpha           = 3
    beta            = 1.5
    decayRate       = 0.3
    evalFunc        = meanSpeedAdaSOTL_v2

    setFlows(1500, 3600, "../2x3.flow.xml")
    
    params = [k, alpha, beta, decayRate]
    stepSizes = [3.0, 0.3, 0.1, 0.1]
    plotFolderPath = "../Plots/HillClimbing_AdaSOTL_v2_2x3_5runs_strat1_1500veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment  
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=10, numRuns=1, strategy=2, paramValidCallbacks=[checkAdaSOTL_v2_params], useGradient=False, reduceStepSize = True)