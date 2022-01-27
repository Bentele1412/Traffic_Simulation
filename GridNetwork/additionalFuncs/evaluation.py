#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

import traci
import os
import numpy as np
from sumolib import checkBinary
from GridNetwork.additionalFuncs.helper import mapLPDetailsToTL, getTLPhaseInfo, getMeanSpeedWaitingTime, createTrafficLights, setFlows, deleteTempFiles
from controllers.CycleBasedTLController import CycleBasedTLController
from controllers.AdaSOTL import AdaSOTL
from controllers.SOTL import SOTL
import time
import shutil

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

SUMO_CONFIG_PATH = "../2x3.sumocfg"
JTRROUTER_CONFIG_PATH = "../2x3.jtrrcfg"
TRIPINFO_PATH = "../tripinfo.xml"
STATISTICS_PATH = "../statistics.xml"
FLOW_PATH = "../2x3.flow.xml"
ROUTES_PATH = "../2x3Routes.xml"


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
    
    timestamp = str(time.time())
    #shutil.copy(SUMO_CONFIG_PATH, SUMO_CONFIG_PATH[:3]+timestamp+SUMO_CONFIG_PATH[3:])
    configPath = os.path.abspath("../"+timestamp+"2x3.sumocfg")
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

    shutil.copy("../2x3.flow.xml", "../"+timestamp+"2x3.flow.xml")
    setFlows(numVehicles, simulationTime, "../"+timestamp+"2x3.flow.xml")
    #os.system('jtrrouter -c ../2x3.jtrrcfg')
    os.system('jtrrouter -n ../2x3net.net.xml --additional-files ../paperVehicle.xml -r ../'+timestamp+'2x3.flow.xml -o ../'+timestamp+'2x3Routes.xml --seed '+timestamp.replace(".", "")[-8:])

    '''
    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "../tripinfo.xml",
                                    "--statistic-output", "../statistics.xml"])
    '''
    traci.start([sumoBinary, "-n", "../2x3net.net.xml", "-r", "../"+timestamp+"2x3Routes.xml", "--additional-files", "../additionals.xml", "--no-step-log", "true",
                                    "--tripinfo-output", "../"+timestamp+"tripinfo.xml",
                                    "--statistic-output", "../"+timestamp+"statistics.xml"])


    _run(adaSotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime("../"+timestamp+"statistics.xml", "../"+timestamp+"tripinfo.xml")
    deleteTempFiles(timestamp)
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