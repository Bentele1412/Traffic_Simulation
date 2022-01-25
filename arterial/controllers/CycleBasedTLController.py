#!/usr/bin/env python

import numpy as np
import traci

class CycleBasedTLController():
    def __init__(self, tl, cycleTime, phaseShift, numPhases, yellowPhaseDuration):
        self.tl = tl
        self.lastStep = 0
        self.currentStep = 0
        self.countYellowSteps = 0
        self.numPhases = numPhases
        self.yellowPhaseDuration = yellowPhaseDuration
        self.setCycle(cycleTime, phaseShift)

    def setCycle(self, cycleTime, phaseShift):
        self.cycleTime = cycleTime
        self.phaseShift = phaseShift
        totalGreenPhaseDuration = self.cycleTime - (self.numPhases/2)*self.yellowPhaseDuration
        self.phaseArr = []
        phases = np.arange(0, self.numPhases+1, 2)
        for lane, phase in zip(self.tl.lanes, phases): #think about if more than 2 phases / 2 lanes 
            greenPhaseLength = int(np.round(totalGreenPhaseDuration * lane.greenPhaseDurationRatio))
            self.phaseArr.append([phase]*greenPhaseLength)
            self.phaseArr.append([phase+1]*self.yellowPhaseDuration)
        self.phaseArr = [item for sublist in self.phaseArr for item in sublist]
        
        #test correct rounding --> add or subtract a phase dependent on possible rounding mistake
        if len(self.phaseArr) < self.cycleTime:
            self.phaseArr = np.concatenate((np.array([0]), self.phaseArr))
        elif len(self.phaseArr) > self.cycleTime:
            self.phaseArr = self.phaseArr[1:].copy()
        if len(self.phaseArr) != self.cycleTime:
            print("False rounding at calculation of green phase length!")
            print(self.phaseArr)
        
        #include phase shift
        self.phaseArr = np.roll(self.phaseArr, self.phaseShift)

    def step(self):
        #Ensure that tl works consistently when cycles are switched
        #first count how long the last yellow phase has been going ...
        if self.lastStep%2 != 0:
            #yellowPhase
            self.countYellowSteps += 1      
        else:
            #greenPhase
            self.countYellowSteps = 0

        #if there is currently a green phase and the new cycle starts with the same green phase 
        # or the yellow phase just before then continue with the green phase
        if self.lastStep % 2 == 0 and (self.lastStep == self.phaseArr[0] or self.lastStep == self.phaseArr[0]+1):
            #self.currentStep = self.phaseArr[0]
            pass
        # in all other cases, a safe switch guard has to ensure, that we have a yellow phase
        # that is exactly 3 steps/seconds long before the new cycle starts

        elif self.countYellowSteps == 0:
            self.currentStep = self.lastStep+1
        elif self.countYellowSteps < self.yellowPhaseDuration:
            self.currentStep = self.lastStep
        elif self.countYellowSteps == self.yellowPhaseDuration:
            if self.phaseArr[0]%2 != 0:
                self.currentStep = (self.phaseArr[0]+1)%self.numPhases
            else:
                self.currentStep = self.phaseArr[0]
        elif self.countYellowSteps > self.yellowPhaseDuration:
            print("error yellowPhase longer than %d seconds" % self.yellowPhaseDuration)

        traci.trafficlight.setPhase(self.tl.id, self.currentStep)
        self.lastStep = self.currentStep
        self.phaseArr = np.roll(self.phaseArr, -1)

        #calc carsOnLane
        for lane in self.tl.lanes:
            lane.updateCarCount()
            lane.runningAvgDynamics.append(lane.runningAvgCoL)