#!/usr/bin/env python

'''
Imports
'''
from helperTwoPhases import *

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
    configPath = os.path.abspath("2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900

    #create instances
    minGreenTime = 20
    maxGreenTime = 60 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    theta = 41
    sotls = []
    for tl in trafficLights:
        sotls.append(SOTL(tl, mu, theta))

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoGui, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])
    
    run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    