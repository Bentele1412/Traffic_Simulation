#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

import traci
from sumolib import checkBinary
import os
import numpy as np
from controllers.CycleBasedTLController import CycleBasedTLController
from GridNetwork.additionalFuncs.helper import mapLPDetailsToTL, getMeanSpeedWaitingTime, getTLPhaseInfo, createTrafficLights, setFlows

def run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
    step = 0
    pathCounter = 0
    cycleBasedTLControllers = []
    while traci.simulation.getMinExpectedNumber() > 0:
        if step % 1200 == 0 and step < 3600:
            mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
            maxNodeUtilization = max([tl.utilization for tl in trafficLights])
            numPhases, yellowPhaseDuration = getTLPhaseInfo()
            cycleTime = int(np.round(ctFactor * ((1.5 * (numPhases/2)*yellowPhaseDuration + 5)/(1 - maxNodeUtilization))))
            pathCounter += 1
            if step == 0:
                for counter, tl in enumerate(trafficLights):
                    cycleBasedTLControllers.append(CycleBasedTLController(tl, cycleTime, phaseShifts[counter], numPhases, yellowPhaseDuration))
            else:
                for counter, controller in enumerate(cycleBasedTLControllers):
                    controller.setCycle(cycleTime, phaseShifts[counter])
        
        for controller in cycleBasedTLControllers:
            controller.step()
        traci.simulationStep()

        step += 1
    traci.close()
    sys.stdout.flush()

if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    sumoGui = checkBinary('sumo-gui')
    configPath = os.path.abspath("../2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900
    ctFactor = 0.9
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0, 50, 100, 50, 100, 150]
    lpSolveResultPaths = ['../LPSolve/2x3Grid_a_eps0,4.lp.csv', '../LPSolve/2x3Grid_b_eps0,4.lp.csv', '../LPSolve/2x3Grid_c_eps0,4.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c ../2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])

    run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    print("Mean speed: ", meanSpeed)
    print("Mean waiting time: ", meanWaitingTime)
    
    