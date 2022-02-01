#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedCycleBased
from GridNetwork.additionalFuncs.helper import checkCTFactor, setFlows
import random

if __name__ == '__main__':
    random.seed(32)
    ctFactor = random.uniform(0.75, 1.5)
    phaseShifts = [random.randint(10, 150), random.randint(10, 150), random.randint(10, 150), random.randint(10, 150), random.randint(10, 150)]
    evalFunc = meanSpeedCycleBased
    setFlows(1200, 3600, "../2x3.flow.xml")
    
    params = [ctFactor] + phaseShifts
    stepSizes = [0.1] + [2]*5
    plotFolderPath = "../Plots/HillClimbing_CB_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.1, maxIter=1, numRuns=1, strategy=1, paramValidCallbacks=[checkCTFactor])