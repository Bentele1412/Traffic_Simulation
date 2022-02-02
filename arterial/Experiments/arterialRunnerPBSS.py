#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

import os
import traci
from sumolib import checkBinary
from controllers.PBSS import PBSS
from arterial.additionalFuncs.helper import getMeanSpeedWaitingTime, createTrafficLights, setFlows_arterial
import numpy as np


def run(pbss):
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for p in pbss:
            p.step(step)
        
    traci.close()
    sys.stdout.flush()


if __name__ == '__main__':
    meanSpeeds = []
    meanWaitingTimes = []
    replications = 5

    for rep in range(replications):
        sumoBinary = checkBinary('sumo')
        sumoGui = checkBinary('sumo-gui')
        configPath = os.path.abspath("../arterial.sumocfg")
        simulationTime = 3600
        numVehicles = 1200

        #create instances
        minGreenTime = 5
        maxGreenTime = 55 
        trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

        pbss = []
        for tl in trafficLights:
            pbss.append(PBSS(tl, useAAC=True, usePBE=True, usePBS=True))

        setFlows_arterial(numVehicles, simulationTime, "../arterial.flow.xml", delta_r_t=2/16)
        os.system('jtrrouter -c ../arterial.jtrrcfg')

        traci.start([sumoBinary, "-c", configPath,
                                        "--tripinfo-output", "../tripinfo.xml",
                                        "--statistic-output", "../statistics.xml"])
        
        run(pbss)

        meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime("../statistics.xml", "../tripinfo.xml")
        meanSpeeds.append(meanSpeed)
        meanWaitingTimes.append(meanWaitingTime)
    print("Mean speed: ", np.mean(meanSpeeds))
    print("Mean waiting time: ", np.mean(meanWaitingTimes))
    
    