#!/usr/bin/env python

import traci
import sys
import os
import numpy as np
from sumolib import checkBinary
from additionalFuncs.helper import mapLPDetailsToTL, getTLPhaseInfo, getMeanSpeedWaitingTime, createTrafficLights, setFlows
from controllers.CycleBasedTLController import CycleBasedTLController
from controllers.AdaSOTL import AdaSOTL
from controllers.SOTL import SOTL

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def meanSpeedCycleBased(params):
    def _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
        step = 0
        pathCounter = 0
        cycleBasedTLControllers = []
        while traci.simulation.getMinExpectedNumber() > 0:
            if step % 1200 == 0 and step < 3600:
                mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
                maxNodeUtilization = max([tl.utilization for tl in trafficLights])
                numPhases, yellowPhaseDuration = getTLPhaseInfo()
                cycleTime = int(np.round(ctFactor * ((1.5 * (numPhases/2)*yellowPhaseDuration + 5)/(1 - maxNodeUtilization)))) #maybe edit hard coded yellow phases and extract them from file
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

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("../2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900
    ctFactor = params[0]
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0] + list(map(lambda x: int(x), params[1:]))
    lpSolveResultPaths = ['../LPSolve/2x3Grid_a_eps0,4.lp.csv', '../LPSolve/2x3Grid_b_eps0,4.lp.csv', '../LPSolve/2x3Grid_c_eps0,4.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c ../2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])

    _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed), float(meanWaitingTime)

def meanSpeedAdaSOTL(params):
    def _run(adaSotls):
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for adasotl in adaSotls:
                adasotl.step()
            step += 1
        traci.close()
        sys.stdout.flush()

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("../2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    
    beta = params[1]
    alpha = params[0]
    adaSotls = []
    for tl in trafficLights:
        adaSotls.append(AdaSOTL(tl, mu, alpha, beta))

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c ../2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    _run(adaSotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed), float(meanWaitingTime)

def meanSpeedSOTL(params):
    def _run(sotls):
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for sotl in sotls:
                sotl.step()
            step += 1
        traci.close()
        sys.stdout.flush()

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("../2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    
    theta = params[0]
    sotls = []
    for tl in trafficLights:
        sotls.append(SOTL(tl, mu, theta))

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c ../2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    
    _run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed), float(meanWaitingTime)