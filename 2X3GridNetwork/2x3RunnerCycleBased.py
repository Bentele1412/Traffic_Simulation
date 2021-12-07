#!/usr/bin/env python

from helperTwoPhases import *

def run(trafficLights, ctFactor, greenPhaseShifts, lpSolveResultPaths):
    step = 0
    pathCounter = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        if step % 1200 == 0:
            mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
            maxNodeUtilization = max([tl.utilization for tl in trafficLights])
            cycleTime = ctFactor * ((1.5 * 2*3 + 5)/(1 - maxNodeUtilization))
            print(cycleTime)
            pathCounter += 1
        traci.simulationStep()

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
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoGui, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])

    ctFactor = 0.9
    greenPhaseShifts = [0]*5 
    lpSolveResultPaths = ['./LPSolve/2x3Grid_a.lp.csv', './LPSolve/2x3Grid_b.lp.csv', './LPSolve/2x3Grid_c.lp.csv']
    run(trafficLights, ctFactor, greenPhaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    