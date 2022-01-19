#!/usr/bin/env python

import traci

class Detector():
    def __init__(self, id):
        self.id = id
        self.lastVehicleIDs = []
        self.detectedVehicles = 0
        self.vehicleIDs = []

    def getCurrentCars(self):
        incomingCars = 0
        self.vehicleIDs = traci.lanearea.getLastStepVehicleIDs(self.id)
        for id in self.lastVehicleIDs:
            if not id in self.vehicleIDs:
                incomingCars += 1
        self.lastVehicleIDs = self.vehicleIDs
        self.detectedVehicles += incomingCars
        return incomingCars

    def resetDetectedVehicles(self):
        self.detectedVehicles = 0