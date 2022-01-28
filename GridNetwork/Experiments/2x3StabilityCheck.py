#!/usr/bin/env python
import sys
sys.path.insert(0, "../../")

import traci
from sumolib import checkBinary
import os
import numpy as np
import matplotlib.pyplot as plt
from controllers.CycleBasedTLController import CycleBasedTLController
from GridNetwork.additionalFuncs.helper import mapLPDetailsToTL, createTrafficLights, setFlows

if __name__ == '__main__':
    def _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
        step = 0
        pathCounter = 0
        cycleBasedTLControllers = []
        while traci.simulation.getMinExpectedNumber() > 0:
            if step % 3601 == 0 and step < 3600:
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
        return cycleBasedTLControllers

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900
    ctFactor = 0.9
    phaseShifts = [0, 10, 20, 10, 20, 30]
    lpSolveResultPaths = ['./LPSolve/2x3Grid_a_eps0,4.lp.csv', './LPSolve/2x3Grid_b_eps0,4.lp.csv', './LPSolve/2x3Grid_c_eps0,4.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime, "../2x3.flow.xml")
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])

    cycleBasedControllers = _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)
    
    plt.plot(cycleBasedControllers[0].tl.lanes[0].runningAvgDynamics)
    plt.show()