#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

from optimizers.HillClimbing import HillClimbing
from GridNetwork.additionalFuncs.evaluation import meanSpeedSOTL
import pickle

if __name__ == '__main__':
    theta = 30
    evalFunc = meanSpeedSOTL
    
    params = [theta]
    stepSizes = [1]
    hillClimbing = HillClimbing(evalFunc, params, stepSizes)
    hillClimbing.optimize(epsilon=0.001, maxIter=50, numRuns=5, strategy=1)
    dynamics = {'meanSpeed': hillClimbing.fitnessDynamics,
                'stdMeanSpeed': hillClimbing.stdMeanSpeeds,
                'meanWaitingTime': hillClimbing.meanWaitingTimes,
                'stdMeanWaitingTime': hillClimbing.stdWaitingTimes}
    with open("Dynamics_HillClimbing_SOTL_strat1_5runs.pickle", 'wb') as f:
        pickle.dump(dynamics, f)