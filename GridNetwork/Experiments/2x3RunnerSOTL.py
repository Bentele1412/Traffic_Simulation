#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

import os
import traci
from sumolib import checkBinary
from controllers.SOTL import SOTL
from GridNetwork.additionalFuncs.helper import getMeanSpeedWaitingTime, createTrafficLights, setFlows


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
    configPath = os.path.abspath("../2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    theta = 30
    sotls = []
    for tl in trafficLights:
        sotls.append(SOTL(tl, mu, theta))

    setFlows(numVehicles, simulationTime, "../2x3.flow.xml")
    os.system('jtrrouter -c ../2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime("../statistics.xml", "../tripinfo.xml")
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    