#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

import os
import traci
from sumolib import checkBinary
from controllers.SOTL import SOTL
from arterial.additionalFuncs.helper import getMeanSpeedWaitingTime, createTrafficLights, setFlows_arterial


def run(sotls):
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for sotl in sotls:
            sotl.step()
        step += 1
    traci.close()
    sys.stdout.flush()


if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    sumoGui = checkBinary('sumo-gui')
    configPath = os.path.abspath("../arterial.sumocfg")
    simulationTime = 3600
    numVehicles = 1200

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    theta = 32
    sotls = []
    for tl in trafficLights:
        sotls.append(SOTL(tl, mu, theta))

    setFlows_arterial(numVehicles, simulationTime)
    os.system('jtrrouter -c ../arterial.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    