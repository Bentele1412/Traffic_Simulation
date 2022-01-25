#!/usr/bin/env python
import sys
sys.path.insert(0, "../")

from optimizers.HillClimbing import HillClimbing
from additionalFuncs.evaluation import meanSpeedAdaSOTL
import pickle

if __name__ == '__main__':
    alpha = 3.5665
    beta = 1.45495
    evalFunc = meanSpeedAdaSOTL
    
    params = [alpha, beta]
    stepSizes = [0.5, 0.05]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.001, maxIter=50, numRuns=5, strategy=1)
    dynamics = {'meanSpeed': hillClimbing.fitnessDynamics,
                'stdMeanSpeed': hillClimbing.stdMeanSpeeds,
                'meanWaitingTime': hillClimbing.meanWaitingTimes,
                'stdMeanWaitingTime': hillClimbing.stdWaitingTimes}
    with open("Dynamics_HillClimbing_AdaSOTL_strat1_5runs_rerun.pickle", 'wb') as f:
        pickle.dump(dynamics, f)