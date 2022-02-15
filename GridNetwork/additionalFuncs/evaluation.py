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
from controllers.AdaSOTL_v2 import AdaSOTL_v2
from controllers.SOTL import SOTL
import time
import shutil
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

NETFILE_PATH = "../2x3net.net.xml"
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
    #timestamp = str(time.time())
    timestamp = str(random.uniform(0.0, 10000.0))
    simulationTime = 3600
    numVehicles = 1200
    ctFactor = params[0]
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0] + list(map(lambda x: int(x), params[1:]))
    lpSolveResultPaths = ['../LPSolve/2x3Grid_a_eps0,2.lp.csv', '../LPSolve/2x3Grid_b_eps0,2.lp.csv', '../LPSolve/2x3Grid_c_eps0,2.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    #shutil.copy(FLOW_PATH, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    #setFlows(numVehicles, simulationTime, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    os.system('jtrrouter -n '+NETFILE_PATH+' --additional-files ../paperVehicle.xml -r '+FLOW_PATH+' -o '+ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:]+' --seed '+timestamp.replace(".", "")[-8:])
    traci.start([sumoBinary, "-n", NETFILE_PATH, "-r", ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:], "--additional-files", "../additionals.xml", "--no-step-log", "true",
                                    "--tripinfo-output", TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:],
                                    "--statistic-output", STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:]])

    _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime(STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:], TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:])
    deleteTempFiles(timestamp)
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
    
    #timestamp = str(time.time())
    #random.seed(12345)
    timestamp = str(random.uniform(0.0, 10000.0))
    #shutil.copy(FLOW_PATH, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    simulationTime = 3600
    numVehicles = 1200

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

    #setFlows(numVehicles, simulationTime, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    os.system('jtrrouter -n '+NETFILE_PATH+' --additional-files ../paperVehicle.xml -r '+FLOW_PATH+' -o '+ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:]+' --seed '+timestamp.replace(".", "")[-8:])
    traci.start([sumoBinary, "-n", NETFILE_PATH, "-r", ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:], "--additional-files", "../additionals.xml", "--no-step-log", "true",
                                    "--tripinfo-output", TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:],
                                    "--statistic-output", STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:]])


    _run(adaSotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime(STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:], TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:])
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
    #timestamp = str(time.time())
    timestamp = str(random.uniform(0.0, 10000.0))
    simulationTime = 3600
    numVehicles = 1200

    #create instances
    minGreenTime = 20
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    
    theta = params[0]
    sotls = []
    for tl in trafficLights:
        sotls.append(SOTL(tl, mu, theta))

    #shutil.copy(FLOW_PATH, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    #setFlows(numVehicles, simulationTime, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    os.system('jtrrouter -n '+NETFILE_PATH+' --additional-files ../paperVehicle.xml -r '+FLOW_PATH+' -o '+ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:]+' --seed '+timestamp.replace(".", "")[-8:])
    traci.start([sumoBinary, "-n", NETFILE_PATH, "-r", ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:], "--additional-files", "../additionals.xml", "--no-step-log", "true",
                                    "--tripinfo-output", TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:],
                                    "--statistic-output", STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:]])
    
    _run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime(STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:], TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:])
    deleteTempFiles(timestamp)
    return float(meanSpeed), float(meanWaitingTime)

def meanSpeedAdaSOTL_v2(params):
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
    
    #timestamp = str(time.time())
    #random.seed(12345)
    timestamp = str(random.uniform(0.0, 10000.0))
    #shutil.copy(FLOW_PATH, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    simulationTime = 3600
    numVehicles = 1200

    #create instances
    minGreenTime = 5
    maxGreenTime = 55 #change to maxRedTime
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)

    mu = 3
    
    k = params[0]
    alpha = params[1]
    beta = params[2]
    decayRate = params[3]
    adaSotls = []
    for tl in trafficLights:
        adaSotls.append(AdaSOTL_v2(tl, mu, k, alpha, beta, decayRate))

    #setFlows(numVehicles, simulationTime, FLOW_PATH[:3]+timestamp+FLOW_PATH[3:])
    os.system('jtrrouter -n '+NETFILE_PATH+' --additional-files ../paperVehicle.xml -r '+FLOW_PATH+' -o '+ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:]+' --seed '+timestamp.replace(".", "")[-8:])
    traci.start([sumoBinary, "-n", NETFILE_PATH, "-r", ROUTES_PATH[:3]+timestamp+ROUTES_PATH[3:], "--additional-files", "../additionals.xml", "--no-step-log", "true",
                                    "--tripinfo-output", TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:],
                                    "--statistic-output", STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:]])


    _run(adaSotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime(STATISTICS_PATH[:3]+timestamp+STATISTICS_PATH[3:], TRIPINFO_PATH[:3]+timestamp+TRIPINFO_PATH[3:])
    deleteTempFiles(timestamp)
    return float(meanSpeed), float(meanWaitingTime)