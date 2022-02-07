#!/usr/bin/env python

class Lane():
    def __init__(self, id, detectors):
        self.id = id
        self.detectors = detectors #[begin, mid, end]

        self.isRed = True
        self.carsOnLane = 0
        self.runningAvgCoL = 0 #for system stability
        self.runningAvgDynamics = [] #for system stability
        self.carsWithinOmega = 0
        self.utilization = 0
        self.inflowRate = 0
        self.outflowRate = 0
        self.greenPhaseDurationRatio = 0

    def updateCarCount(self, decayRate = 0.1):
        incomingCars = self.detectors[0].getCurrentCars()
        incomingCarsOmega = self.detectors[1].getCurrentCars()
        outflowingCars = self.detectors[-1].getCurrentCars()
        
        self.carsOnLane += incomingCars - outflowingCars
        self.carsWithinOmega += incomingCarsOmega - outflowingCars
        self.runningAvgCoL = (1 - decayRate)*self.runningAvgCoL + decayRate*self.carsOnLane