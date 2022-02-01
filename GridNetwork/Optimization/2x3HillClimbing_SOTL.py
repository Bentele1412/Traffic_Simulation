#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedSOTL
from GridNetwork.additionalFuncs.helper import setFlows
import random

if __name__ == '__main__':
    random.seed(32)
    theta = random.randint(20, 100)
    evalFunc = meanSpeedSOTL
    setFlows(1200, 3600, "../2x3.flow.xml")
    
    params = [theta]
    stepSizes = [1]
    plotFolderPath = "../Plots/HillClimbing_SOTL_2x3_5runs_strat1_1200veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=100, numRuns=5, strategy=1, useGradient=False)