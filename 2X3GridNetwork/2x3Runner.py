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
        vehicleIDs = traci.inductionloop.getLastStepVehicleIDs(self.id)
        for id in vehicleIDs:
            if not id in self.lastVehicleIDs:
                self.detectedVehicles += 1
        self.lastVehicleIDs = vehicleIDs

    def resetDetectedVehicles(self):
        self.detectedVehicles = 0

class Lane():
    def __init__(self, id, detectors, correspondingTLPhase):
        self.id = id
        self.detectors = detectors
        self.correspondingTLPhase = correspondingTLPhase

        self.isRed = True