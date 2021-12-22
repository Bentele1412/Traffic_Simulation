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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


'''
Class definitions
'''

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
        self.utilization = 0

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

class Lane():
    def __init__(self, id, detectors):
        self.id = id
        self.detectors = detectors #[begin, mid, end]

        self.isRed = True
        self.carsOnLane = 0
        self.carsWithinOmega = 0
        self.utilization = 0
        self.inflowRate = 0
        self.outflowRate = 0
        self.greenPhaseDurationRatio = 0

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
        self.phi_min = self.tl.minGreenTime
        self.phi = 0

    def step(self):
        currentPhase = self.tl.getCurrentPhase()
        if currentPhase % 2 == 0: #don´t execute SOTL if TL in yellow phase
            self.phi += 1
        for lane in self.tl.lanes:
            lane.updateCarCount()
            if lane.isRed: 
                self.kappa += lane.carsOnLane #change to more kappas if more than one direction has red
        if self.phi >= self.phi_min:
            for lane in self.tl.lanes:
                if not lane.isRed:
                    if not(0 < lane.carsWithinOmega and lane.carsWithinOmega < self.mu) or self.phi > self.tl.maxGreenTime:
                        if self.kappa >= self.theta:
                            self.tl.switchLight(self.tl.getCurrentPhase())
                            self.resetParams()
                            break
    
    def resetParams(self):
        self.phi = 0
        self.kappa = 0


class CycleBasedTLController():
    def __init__(self, tl, cycleTime, phaseShift):
        self.tl = tl
        self.cycleTime = cycleTime
        self.phaseShift = phaseShift

        totalGreenPhaseDuration = self.cycleTime - 2*3 #maybe remove hard coded yellow phase durations
        self.phaseArr = []
        phases = [0, 2]
        for lane, phase in zip(self.tl.lanes, phases): #think about if more than 2 phases / 2 lanes 
            greenPhaseLength = int(np.round(totalGreenPhaseDuration * lane.greenPhaseDurationRatio))
            self.phaseArr.append([phase]*greenPhaseLength)
            self.phaseArr.append([phase+1]*3) #hard coded yellow phase durations
        self.phaseArr = [item for sublist in self.phaseArr for item in sublist]
        
        #test correct rounding
        if len(self.phaseArr) != self.cycleTime:
            self.phaseArr = np.concatenate((np.array([0]), self.phaseArr))
            if len(self.phaseArr) != self.cycleTime:
                print("False rounding at calculation of green phase length!")
                print(self.phaseArr)
        
        #include phase shift
        self.phaseArr = np.roll(self.phaseArr, self.phaseShift)

    def step(self):
        traci.trafficlight.setPhase(self.tl.id, self.phaseArr[0])
        self.phaseArr = np.roll(self.phaseArr, -1)

    
class HillClimbing():
    def __init__(self, evalFunc, params, stepSizes):
        self.evalFunc = evalFunc
        self.params = params
        self.stepSizes = stepSizes

        self.posDirection = 1
        self.negDirection = -1
        self.fitness = self.evalFunc(self.params)
    
    def optimize(self, epsilon=1, numRuns=1, maxIter=50):
        self.epsilon = epsilon
        self.numRuns = numRuns
        print("NumRuns: ", self.numRuns)
        print("maxIter: ", maxIter)

        self.fitnessDynamics = [self.fitness]

        for i in range(maxIter):
            gradient = self._calcGradient()
            print("Iteration %i done." % (i+1))
            print(gradient)
            print(np.linalg.norm(gradient))
            if any(gradient): #and np.linalg.norm(gradient) > self.epsilon:
                self.fitness = self.fitness + max(gradient)
                self.params = self.params+gradient*self.stepSizes
                self.fitnessDynamics.append(self.fitness)
            else:
                break
        print("Found optimum with:")
        print("Optimal fitness:", self.fitness)
        print("Optimal params:", self.params)
        
        plt.plot(self.fitnessDynamics)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Fitness dynamics")
        plt.show()

    def _calcGradient(self):
        fitnessDevResults = []
        for i in range(len(self.params)):
            direction = np.zeros(len(self.params))
            direction[i] = self.posDirection
            posParams = self.params+direction*self.stepSizes
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            fitnessDevResults.append(self._performRuns(posParams) - self.fitness) 
            fitnessDevResults.append(self._performRuns(negParams) - self.fitness)
        maxInd = np.argmax(fitnessDevResults)
        if maxInd % 2 == 0:
            maxFitnessDev = fitnessDevResults[maxInd]
        else:
            maxFitnessDev = -fitnessDevResults[maxInd]
        gradient = np.zeros(len(self.params))
        gradient[int(np.floor(maxInd/2))] = maxFitnessDev
        return gradient

    def _performRuns(self, params):
        fitnesses = []
        for _ in range(self.numRuns):
            fitnesses.append(self.evalFunc(params))
        return np.mean(fitnesses)
            
'''
Evaluation functions
'''
def meanSpeedCycleBased(params):
    def _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
        step = 0
        pathCounter = 0
        cycleBasedTLControllers = []
        while traci.simulation.getMinExpectedNumber() > 0:
            if step % 1200 == 0 and step < 3600:
                mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
                maxNodeUtilization = max([tl.utilization for tl in trafficLights])
                cycleTime = int(np.round(ctFactor * ((1.5 * 2*3 + 5)/(1 - maxNodeUtilization)))) #maybe edit hard coded yellow phases and extract them from file
                pathCounter += 1
                for counter, tl in enumerate(trafficLights):
                    cycleBasedTLControllers.append(CycleBasedTLController(tl, cycleTime, phaseShifts[counter]))
            
            for controller in cycleBasedTLControllers:
                controller.step()
            traci.simulationStep()

            step += 1
        traci.close()
        sys.stdout.flush()

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900
    ctFactor = params[0]
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0] + list(map(lambda x: int(x), params[1:]))
    lpSolveResultPaths = ['./LPSolve/2x3Grid_a_eps0,4.lp.csv', './LPSolve/2x3Grid_b_eps0,4.lp.csv', './LPSolve/2x3Grid_c_eps0,4.lp.csv']

    #create instances
    trafficLights = createTrafficLights()

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])

    _run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed)

'''
Helper functions
'''

def getMeanSpeedWaitingTime():
    tree = ET.parse("statistics.xml")
    root = tree.getroot()
    avgSpeed = root.find('vehicleTripStatistics').attrib['speed']
    avgWaitingTime = root.find('vehicleTripStatistics').attrib['waitingTime']
    return avgSpeed, avgWaitingTime

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
    for counter, detector in enumerate(root.iter("e2Detector")):
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

def createTrafficLights(minGreenTime = 5, maxGreenTime = 60):
    '''
    Create list of all TrafficLights containing all corresponding lanes which consist of their corresponding detectors 
    '''

    trafficLights = []
    lanes = []
    detectors = []
    tree = ET.parse("additionals.xml")
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
    tree = ET.parse("2x3.flow.xml")
    root = tree.getroot()
    for counter, flow in enumerate(root.iter("flow")):
        flow.set("probability", str(probabilities[counter]))
    tree.write("2x3.flow.xml")

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