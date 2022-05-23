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
    setFlows(1500, 3600, "../2x3.flow.xml")
    
    params = [ctFactor] + phaseShifts
    stepSizes = [0.05] + [2]*5
    plotFolderPath = "../Plots/HillClimbing_CB_2x3_5runs_strat1_1500veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=50, numRuns=5, strategy=1, paramValidCallbacks=[checkCTFactor], useGradient=False, reduceStepSize = True)