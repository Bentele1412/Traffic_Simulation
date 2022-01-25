#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")

import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
from sumoadditionals.Detector import Detector
from sumoadditionals.Lane import Lane
from sumoadditionals.TrafficLight import TrafficLight

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def getTLPhaseInfo():
    tree = ET.parse("../2x3net.net.xml")
    root = tree.getroot()
    tls = root.find('tlLogic')
    numPhases = len(tls)
    yellowPhaseDurations = tls[1].attrib['duration']
    return numPhases, int(yellowPhaseDurations)

def getMeanSpeedWaitingTime():
    tree = ET.parse("../statistics.xml")
    root = tree.getroot()
    avgSpeed = root.find('vehicleTripStatistics').attrib['speed']
    avgWaitingTime = root.find('vehicleTripStatistics').attrib['waitingTime']
    return avgSpeed, avgWaitingTime

def createTrafficLights(minGreenTime = 5, maxGreenTime = 60):
    '''
    Create list of all TrafficLights containing all corresponding lanes which consist of their corresponding detectors 
    '''

    trafficLights = []
    lanes = []
    detectors = []
    tree = ET.parse("../additionals.xml")
    root = tree.getroot()
    for counter, detector in enumerate(root.iter("e2Detector")):
        if counter % 3 == 0 and counter != 0:
            lanes.append(Lane(prevDetector.get('lane'), detectors))
            detectors = []
        if counter % 6 == 0 and counter != 0:
            trafficLights.append(TrafficLight(prevDetector.get('lane')[2:4], lanes, minGreenTime, maxGreenTime))
            lanes = []
        detectors.append(Detector(detector.get('id')))
        prevDetector = detector
    lanes.append(Lane(prevDetector.get('lane'), detectors))
    trafficLights.append(TrafficLight(prevDetector.get('lane')[2:4], lanes, minGreenTime, maxGreenTime))
    return trafficLights

def setFlows(numVehicles, simulationTime):
    groundProb = numVehicles/simulationTime/12
    heavyProb = groundProb*7
    probabilities = [groundProb]*5
    for _ in range(3):
        probabilities.append(heavyProb)
    tree = ET.parse("../2x3.flow.xml")
    root = tree.getroot()
    for counter, flow in enumerate(root.iter("flow")):
        flow.set("probability", str(probabilities[counter]))
    tree.write("../2x3.flow.xml")

def setFlows_arterial(numVehicles, simulationTime, delta_r_t=1/16):
    groundProb = numVehicles/simulationTime/16
    
    probabilities = [groundProb]*4          #vertical flows
    probabilities.append(groundProb*7)      # horizontal flows

    #flows with turn probability
    prob_turners_total = 5*groundProb # = r_s + r_t
    r_t = 0
    for _ in range(3):
        r_s = prob_turners_total - r_t
        probabilities.append(r_s) # vehicles going straight at intersection O
        if r_t != 0:
            probabilities.append(r_t) # vehicles that turn at intersection O
        r_t += groundProb*delta_r_t*16
    tree = ET.parse("../arterial.flow.xml")
    root = tree.getroot()
    for counter, flow in enumerate(root.iter("flow")):
        flow.set("probability", str(probabilities[counter]))
    tree.write("../arterial.flow.xml")

def mapLPDetailsToTL(trafficLights, path):
    lpSolveResults = pd.read_csv(path, sep=';')
    lpTrafficLightIds = np.arange(1, len(trafficLights)+1, 1) #tl sorted in correct structure (from north-west to north-east and then from south-west to south-east)
    lpLaneDirections = ["1A", "3C"] #A = north, C = west #lanes ordered like north, west
    for trafficLight, lpID in zip(trafficLights, lpTrafficLightIds):
        lpID = str(lpID)
        sumUtilization = 0
        for lane, lpLaneDirection in zip(trafficLight.lanes, lpLaneDirections):
            utilizationRow = lpSolveResults[lpSolveResults['Variables'] == "u" + lpID + "_" + lpLaneDirection]
            lane.utilization = utilizationRow['result'].values[0]
            lane.utilization = float(lane.utilization.replace(',', '.'))
            sumUtilization += lane.utilization
            inflowRateRow = lpSolveResults[lpSolveResults['Variables'] == "i" + lpID + lpLaneDirection[-1]]
            lane.inflowRate = inflowRateRow['result'].values[0]
            lane.inflowRate = float(lane.inflowRate.replace(',', '.'))
            outFlowRateRow = lpSolveResults[lpSolveResults['Variables'] == "o" + lpID + lpLaneDirection[-1]]
            lane.outflowRate = outFlowRateRow['result'].values[0]
            lane.outflowRate = float(lane.outflowRate.replace(',', '.'))
        
        trafficLight.utilization = sumUtilization
        for lane in trafficLight.lanes:
            lane.greenPhaseDurationRatio = lane.utilization/trafficLight.utilization

def checkCTFactor(params):
    ctFactor = params[0]
    if ctFactor < 0.75:
        params[0] = 0.75
    elif ctFactor > 1.5:
        params[0] = 1.5
    return params