#!/usr/bin/env python

from helperTwoPhases import *

def run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
    step = 0
    pathCounter = 0
    cycleBasedTLControllers = []
    while traci.simulation.getMinExpectedNumber() > 0:
        if step % 1200 == 0 and step < 3600:
            mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
            maxNodeUtilization = max([tl.utilization for tl in trafficLights])
            cycleTime = int(np.round(ctFactor * ((1.5 * 2*3 + 5)/(1 - maxNodeUtilization)))) #maybe edit hard coded yellow phases and extract them from file
            pathCounter += 1
            for counter, tl in enumerate(trafficLights):
                cycleBasedTLControllers.append(CycleBasedTLController(tl, cycleTime, phaseShifts[counter]))
        
        for controller in cycleBasedTLControllers:
            controller.step()
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
    ctFactor = 0.9
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0, 5, 10, 5, 10, 15]
    lpSolveResultPaths = ['./LPSolve/2x3Grid_a_eps0,4.lp.csv', './LPSolve/2x3Grid_b_eps0,4.lp.csv', './LPSolve/2x3Grid_c_eps0,4.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoGui, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])

    run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    