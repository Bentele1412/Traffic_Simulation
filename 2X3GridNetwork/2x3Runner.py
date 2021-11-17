#!/usr/bin/env python

'''
Imports
'''
from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
from sumolib import checkBinary
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


'''
Helper functions
'''

def getMeanSpeedWaitingTime():
    tree = ET.parse("statistics.xml")
    root = tree.getroot()
    avgSpeed = root.find('vehicleTripStatistics').attrib['speed']
    avgWaitingTime = root.find('vehicleTripStatistics').attrib['waitingTime']
    return avgSpeed, avgWaitingTime

class TrafficLight():
    '''
    General Notes:
    - One simulation step is 1 second.
    - Phase 0 of every traffic light means vertical lane has green.
    - Phase 2 of every traffic light means horizontal lane has green.
    '''
    def __init__(self, trafficLightId: str, lanes: list, minGreenTime: float, maxGreenTime: float):
        '''
        Parameters:
        -----------
        trafficLightId: str
            Net id of corresponding traffic light.
        lanes: list
            List of incoming lane ids connected to traffic light.
            sorted --> [lane from north, lane from west]
        detectors: list
            List of detectors placed on incoming lanes of traffic light.
            sorted --> [lane from north, lane from west]
        minGreenTime: float
            Minimum green time for each green phase in seconds.
        maxGreenTime: float
            Maximum green time for each green phase in seconds.

        Returns:
        --------
        None.
        '''

        self.id = trafficLightId
        self.lanes = lanes 
        self.lanes[0].isRed = False #first phase is always 0 --> lane coming from north has green
        self.minGreenTime = minGreenTime
        self.maxGreenTime = maxGreenTime

        self.carsApproachingRed = 0

    def switchLight(self, currentPhase):
        traci.trafficlight.setPhase(self.id, currentPhase + 1)
        self.carsApproachingRed = 0
        self.toggleLanePhases()

    def getCurrentPhase(self):
        return traci.trafficlight.getPhase(self.id)

    def toggleLanePhases(self):
        for lane in self.lanes:
            lane.isRed = not lane.isRed

class Detector():
    def __init__(self, id):
        self.id = id
        self.lastVehicleIDs = []
        self.detectedVehicles = 0

    def getCurrentCars(self):
        incomingCars = 0
        vehicleIDs = traci.inductionloop.getLastStepVehicleIDs(self.id)
        for id in vehicleIDs:
            if not id in self.lastVehicleIDs:
                incomingCars += 1
        self.lastVehicleIDs = vehicleIDs
        self.detectedVehicles += incomingCars
        return incomingCars

    def resetDetectedVehicles(self):
        self.detectedVehicles = 0

class Lane():
    def __init__(self, id, detectors):
        self.id = id
        self.detectors = detectors #[begin, mid, end]

        self.isRed = True
        self.carsOnLane = 0
        self.carsWithinOmega = 0

    def updateCarCount(self):
        incomingCars = self.detectors[0].getCurrentCars()
        incomingCarsOmega = self.detectors[1].getCurrentCars()
        outflowingCars = self.detectors[-1].getCurrentCars()
        
        self.carsOnLane += incomingCars - outflowingCars
        self.carsWithinOmega += incomingCarsOmega - outflowingCars

class SOTL():
    def __init__(self, tl, mu, theta):
        self.tl = tl
        self.mu = mu
        self.theta = theta

        self.kappa = 0
        self.phi_min = self.tl.minGreenPhase
        self.phi = 0

    def step(self):
        self.phi += 1
        for lane in self.tl.lanes:
            if lane.isRed: 
                self.kappa += lane.carsOnLane
                break
        if self.phi >= self.phi_min:
            for lane in self.tl.lanes:
                if not lane.isRed:
                    if not(0 < lane.carsWithinOmega and lane.carsWithinOmega < self.mu):
                        if self.kappa >= self.theta:
                            self.tl.switchLight(self.tl.getCurrentPhase())
                            self.kappa = 0
                            self.phi = 0
                            break
        

def createDetectors():
    '''
    !Legacy function!
    '''

    # [[[], []]] --> junction, lane, detectors
    detectors = []
    junction = []
    lane = []
    tree = ET.parse("additionals.xml")
    root = tree.getroot()
    for counter, detector in enumerate(root.iter("e1Detector")):
        if counter % 3 == 0 and counter != 0:
            junction.append(lane)
            lane = []
        if counter % 6 == 0 and counter != 0:
            detectors.append(junction)
            junction = []
        lane.append(Detector(detector.get('id')))
    junction.append(lane)
    detectors.append(junction)
    return detectors

def createTrafficLights(minGreenTime, maxGreenTime):
    '''
    Create list of all TrafficLights containing all corresponding lanes which consist of their corresponding detectors 
    '''

    trafficLights = []
    lanes = []
    detectors = []
    tree = ET.parse("additionals.xml")
    root = tree.getroot()
    for counter, detector in enumerate(root.iter("e1Detector")):
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

if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("2x3.sumocfg")

    #create instances
    minGreenTime = 5
    maxGreenTime = 30
    trafficLights = createTrafficLights(minGreenTime, maxGreenTime)
    