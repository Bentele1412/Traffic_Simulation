#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

import os
import traci
from sumolib import checkBinary
from controllers.PBSS import PBSS
from arterial.additionalFuncs.helper import getMeanSpeedWaitingTime, createTrafficLights, setFlows_arterial


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

    setFlows_arterial(numVehicles, simulationTime)
    os.system('jtrrouter -c ../arterial.jtrrcfg')

    traci.start([sumoGui, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    run(pbss)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", float(meanSpeed))
    print("Mean waiting time: ", meanWaitingTime)
    
    