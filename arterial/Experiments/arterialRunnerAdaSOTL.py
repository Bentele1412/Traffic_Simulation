#!/usr/bin/env python
import sys
sys.path.insert(0, "../")

'''
Imports
'''
import os
import traci
from sumolib import checkBinary
from controllers.AdaSOTL import AdaSOTL
from additionalFuncs.helper import getMeanSpeedWaitingTime, createTrafficLights, setFlows_arterial

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


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
    numVehicles = 900

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    beta = 1.18
    alpha = 4
    sotls = []
    for tl in trafficLights:
        sotls.append(AdaSOTL(tl, mu, alpha, beta))

    setFlows_arterial(numVehicles, simulationTime)
    os.system('jtrrouter -c ../arterial.jtrrcfg')

    traci.start([sumoGui, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    