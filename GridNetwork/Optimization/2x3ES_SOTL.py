#!/usr/bin/env python
import sys
sys.path.insert(0, "../")

from optimizers.ES_MuSlashMuCommaLambda import ES_MuSlashMuCommaLambda
from additionalFuncs.evaluation import meanSpeedSOTL
import random
import pickle

if __name__ == '__main__':
    random.seed(12345)
    theta = random.randint(20, 70)
    evalFunc = meanSpeedSOTL
    
    params = [theta]
    mu = 3
    lambda_ = 6
    es = ES_MuSlashMuCommaLambda(params, mu, lambda_)
    es.optimize(evalFunc, isMaximization=True, sigma=1, numRuns=5, maxIter=50)
    dynamics = {'meanSpeed': es.fitnessDynamics,
                'sigmaDynamics': es.sigmaDynamics,
                'stdMeanSpeed': es.stdMeanSpeeds,
                'meanWaitingTime': es.meanWaitingTimes,
                'stdMeanWaitingTime': es.stdWaitingTimes}
    with open("Dynamics_ES_SOTL_mu3_lambda6_5runs.pickle", 'wb') as f:
        pickle.dump(dynamics, f)