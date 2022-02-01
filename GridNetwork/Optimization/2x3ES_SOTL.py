#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.ES_MuSlashMuCommaLambda import ES_MuSlashMuCommaLambda
from GridNetwork.additionalFuncs.evaluation import meanSpeedSOTL
from GridNetwork.additionalFuncs.helper import setFlows
import random

if __name__ == '__main__':
    random.seed(32)
    theta = random.randint(20, 70)
    evalFunc = meanSpeedSOTL
    setFlows(1200, 3600, "../2x3.flow.xml")
    
    params = [theta]
    mu = 3
    lambda_ = 6
    plotFolderPath = "../Plots/Test/" #CAUTION!!!:change before running --> create new folder for each optimization experiment  
    es = ES_MuSlashMuCommaLambda(params, mu, lambda_)
    es.optimize(evalFunc, plotFolderPath= plotFolderPath, isMaximization=True, sigma=1, numRuns=3, maxIter=1)