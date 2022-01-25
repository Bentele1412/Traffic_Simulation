#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

from optimizers.ES_MuSlashMuCommaLambda import ES_MuSlashMuCommaLambda
from GridNetwork.additionalFuncs.evaluation import meanSpeedAdaSOTL

if __name__ == '__main__':
    alpha = 3
    beta = 1.4
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    mu = 2
    lambda_ = 6
    plotFolderPath = "../Plots/ES_AdaSOTL_2x3_5runs_2mu_6lambda_900veh/" #CAUTION!!!:change before running --> create new folder for each optimization experiment  
    es = ES_MuSlashMuCommaLambda(params, mu, lambda_)
    es.optimize(evalFunc, plotFolderPath=plotFolderPath, isMaximization=True, sigma=1, numRuns=5, maxIter=100)