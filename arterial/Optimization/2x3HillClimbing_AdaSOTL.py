#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from arterial.additionalFuncs.evaluation import meanSpeedAdaSOTL

if __name__ == '__main__':
    alpha = 3
    beta = 1.4
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    plotFolderPath = "../Plots/HillClimbing_AdaSOTL_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=100, numRuns=5, strategy=1)