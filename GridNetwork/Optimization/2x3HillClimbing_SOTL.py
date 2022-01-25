#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedSOTL

if __name__ == '__main__':
    theta = 30
    evalFunc = meanSpeedSOTL
    
    params = [theta]
    stepSizes = [1]
    plotFolderPath = "../Plots/HillClimbing_SOTL_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=50, numRuns=5, strategy=1)