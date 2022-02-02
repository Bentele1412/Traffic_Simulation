#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from arterial.additionalFuncs.evaluation import meanSpeedSOTL
from arterial.additionalFuncs.helper import setFlows_arterial
import random

if __name__ == '__main__':
    random.seed(32)
    theta = random.randint(20, 70)
    evalFunc = meanSpeedSOTL
    setFlows_arterial(1200, 3600, "../arterial.flow.xml", delta_r_t=0)
    
    params = [theta]
    stepSizes = [1]
    plotFolderPath = "../Plots/HillClimbing_SOTL_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.01, maxIter=3, numRuns=1, strategy=1)