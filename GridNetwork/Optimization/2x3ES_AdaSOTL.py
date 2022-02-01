#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.ES_MuSlashMuCommaLambda import ES_MuSlashMuCommaLambda
from GridNetwork.additionalFuncs.evaluation import meanSpeedAdaSOTL
from GridNetwork.additionalFuncs.helper import setFlows
import random


if __name__ == '__main__':
    random.seed(32)
    alpha = random.randint(1, 6)
    beta = random.uniform(1.0, 2.0)
    evalFunc = meanSpeedAdaSOTL
    setFlows(1200, 3600, "../2x3.flow.xml")
    
    params = [alpha, beta]
    mu = 3
    lambda_ = 8
    plotFolderPath = "../Plots/ES_AdaSOTL_2x3_5runs_3mu_8lambda_1200veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment  
    es = ES_MuSlashMuCommaLambda(params, mu, lambda_)
    es.optimize(evalFunc, plotFolderPath=plotFolderPath, isMaximization=True, sigma=1, numRuns=5, maxIter=50)