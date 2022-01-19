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
    configPath = os.path.abspath("2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900

    #create instances
    minGreenTime = 5
    maxGreenTime = 55 
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    pbss = []
    for tl in trafficLights:
        pbss.append(PBSS(tl, useAAC=False, usePBE=True, usePBS=False))

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])
    
    run(pbss)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime/3.33)
    
    