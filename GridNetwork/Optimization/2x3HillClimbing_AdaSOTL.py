#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedAdaSOTL

if __name__ == '__main__':
    alpha = 3.5665
    beta = 1.45495
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    plotFolderPath = "../Plots/Test/" #CAUTION!!!:change before running --> create new folder for each optimization experiment  
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=1, numRuns=1, strategy=1)