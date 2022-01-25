#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedCycleBased
from GridNetwork.additionalFuncs.helper import checkCTFactor

if __name__ == '__main__':
    ctFactor = 0.6
    phaseShifts = [10, 20, 10, 20, 30]
    evalFunc = meanSpeedCycleBased
    
    params = [ctFactor] + phaseShifts
    stepSizes = [0.1] + [1]*5
    plotFolderPath = "../Plots/HillClimbing_CB_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.1, maxIter=1, numRuns=1, strategy=1, paramValidCallbacks=[checkCTFactor])