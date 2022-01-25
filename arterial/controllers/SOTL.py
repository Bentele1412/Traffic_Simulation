#!/usr/bin/env python

from modulefinder import Module


class SOTL():
    def __init__(self, tl, mu, theta):
        self.tl = tl
        self.mu = mu
        self.theta = theta

        self.kappas = [0]*(len(self.tl.lanes)-1)
        self.phi_min = self.tl.minGreenTime
        self.phi = 0

    def step(self):
        currentPhase = self.tl.getCurrentPhase()
        if currentPhase % 2 == 0: #don´t execute SOTL if TL in yellow phase
            self.phi += 1
        kappaCounter = 0
        for lane in self.tl.lanes:
            lane.updateCarCount()
            if lane.isRed: 
                self.kappas[kappaCounter] += lane.carsOnLane #change to more kappas if more than one direction has red
                kappaCounter += 1
        if self.phi >= self.phi_min:
            for counter, lane in enumerate(self.tl.lanes):
                if not lane.isRed:
                    if not(0 < lane.carsWithinOmega and lane.carsWithinOmega < self.mu) or self.phi > self.tl.maxGreenTime:
                        if self.kappas[0] >= self.theta: #index out of bounds with self.kappas[counter]
                            self.tl.switchLight(self.tl.getCurrentPhase())
                            self.resetParams()
                            break
    
    def resetParams(self):
        self.phi = 0
        self.kappa = [0]*(len(self.tl.lanes)-1)