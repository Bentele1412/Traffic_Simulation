#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from arterial.additionalFuncs.evaluation import meanSpeedAdaSOTL
from arterial.additionalFuncs.helper import setFlows_arterial
import random

if __name__ == '__main__':
    random.seed(32)
    alpha = random.uniform(1.0, 6.0)
    beta = random.uniform(1.0, 2.0)
    evalFunc = meanSpeedAdaSOTL
    setFlows_arterial(900, 3600, "../arterial.flow.xml")
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    plotFolderPath = "../Plots/HillClimbing_AdaSOTL_2x3_5runs_strat1_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(plotFolderPath=plotFolderPath, epsilon=0.001, maxIter=100, numRuns=5, strategy=1)