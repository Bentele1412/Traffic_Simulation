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
import time


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
        self.runningAvgCoL = 0 #for system stability
        self.runningAvgDynamics = [] #for system stability
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
        self.runningAvgCoL = 0.9*self.runningAvgCoL + (1-0.9)*self.carsOnLane

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

class AdaSOTL():
    '''
    Adaptive threshold self-organizing traffic light controller

    Problem of basic SOTL:
        - Green phases get shorter if traffic load gets heavier but should get longer
        - leads to more yellow phases and crossing clearing --> less time for queue clearing --> lost time for every crossway participant to move on

    Idea:
        - adaptive threshold of integrated cars on lane
        - the more cars driving to a red light of the crossway per lane, the bigger the threshold gets (biggest threshold for all (red) lanes at the crossway)

    Implementation strategy:
        - sum up all cars driving towards crossway and multiply with a constant (constant to be optimized)
        --> threshold =  avgCarsTowardsCrossway**self.beta * self.alpha
        - alpha 
            - constant to adapt threshold to needed scale range
        - beta 
            - exponent to ensure increasing threshold of running averaged cars towards crossway
            - otherwise kappa overshooting theta would need always the same time (linear dependency of carsTowardsCrossway and alpha)

    Open Questions: 
        - Adaptation strength
    '''
    def __init__(self, tl, mu, alpha, beta):
            self.tl = tl
            self.mu = mu
            self.theta = 0
            self.alpha = alpha
            self.beta = beta

            self.kappa = 0
            self.phi_min = self.tl.minGreenTime
            self.phi = 0

    def step(self):
        avgCarsTowardsCrossway = 0
        currentPhase = self.tl.getCurrentPhase()
        if currentPhase % 2 == 0: #don´t execute SOTL if TL in yellow phase
            self.phi += 1
        for lane in self.tl.lanes:
            lane.updateCarCount()
            if lane.isRed: 
                self.kappa += lane.carsOnLane #change to more kappas if more than one direction has red
            
            #adaptivity
            avgCarsTowardsCrossway += lane.runningAvgCoL
        self.theta = avgCarsTowardsCrossway**self.beta * self.alpha

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

class PBSS():
    class Cluster():
        def __init__(self, tau_ps, n_pc, L_det, v_f, tau_pd=1):
            '''
            status: 
                0 = Tuple
                1 = Minor
                2 = Queue
                3 = Platoon
            '''
            self.tau_ps = tau_ps #sample period start time
            self.tau_pd = tau_pd #sample period duration
            self.calcTau_pe() #calc sample period end time
            self.n_pc = n_pc #number of counted vehicles (drove over detector)
            self.calcQ_pc() #calc flow rate of vehicles
            self.L_det = L_det #length between intersection and detector
            self.v_f = v_f #speed
            self.tau_ps_t = self._getTau_ps_t(self.tau_ps)
            self.status = 0 #tuple
        
        def _getTau_ps_t(self, t):
            return self.tau_ps - t + self.L_det/self.v_f

        def calcTau_pe(self):
            self.tau_pe = self.tau_ps_t + self.tau_pd

        def calcQ_pc(self):
            self.q_pc = self.n_pc/self.tau_pd #flow rate of vehicles

    def __init__(self, tl, tau_g_min=5, tau_g_max=55, tau_y=5, tau_sl=3, tau_sh=3, v_f=9.5, c_thc=5, c_thpc=5, useAAC=True, usePBE=True, usePBS=True):
        self.tl = tl
        self.tau_g_min = tau_g_min #min green time
        self.tau_g_max = tau_g_max #max green time
        self.tau_y = tau_y #yellow time
        self.tau_sl = tau_sl #startup loss time
        self.tau_sh = tau_sh #saturation headway
        self.v_f = v_f #0.95*v_max --> here v_max = 10 m/s
        self.c_thc = 0 if useAAC and not usePBE and not usePBS else c_thc #specified threshold duration for creating clusters
        self.c_thpc = c_thpc #threshold of platoon count (in cars)
        self.c_thpd = 1/self.c_thc if self.c_thc != 0 else 0 #flow rate threshold
        self.useAAC = useAAC 
        self.usePBE = usePBE
        self.usePBS = usePBS

        self.S = [[]*len(self.tl.lanes)]
        self.tau_ge = 0

    def step(self, t):
        #don´t save tuple in self.S if n_pc = 0 
        self.tau_ge += 1

    def performAAC(self, tau_ge, s):
        tau_gremain = self.tau_g_max - tau_ge
        n_q = self.calcN_q(s)
        n_qa = self.estimateN_qa(s, tau_ge, tau_adv=0, n_qn=n_q)
        tau_ext = self.getTau_qc(n_qa, tau_ge)
        tau_ext = min(tau_ext, tau_gremain)
        return tau_ext

    def performPBE(self, tau_ge, s_r, s_g):
        tau_gremain = self.tau_g_max - tau_ge
        n_q = self.calcN_q(s_r)
        n_qa_r = self.estimateN_qa(s_r, tau_ge=0, tau_adv=self.tau_y, n_qn=n_q)
        tau_qa_r = self.getTau_qc(n_qa_r, 0)
        tau_qa_r = max(tau_qa_r, tau_gremain)
        platoons_g = [c for c in s_g if c.status == 3]
        if not len(platoons_g) == 0:
            n_m_g = sum([c.n_pc for c in s_g if c.status == 1 and c.tau_ps_t < platoons_g[0].tau_ps_t])
            tau_idle_g = platoons_g[0].tau_ps_t - self.getTau_qc(n_m_g, 0)
            delta_tau = (tau_qa_r + 2 * self.tau_y) - tau_idle_g
            if delta_tau > 0:
                n_m_r = sum([c.n_pc for c in s_r if c.status == 1])
                n_m_r = n_m_r + n_q - n_qa_r #questionable usage of n_q
                delta_r = n_m_r * delta_tau - n_qa_r * (platoons_g[0].tau_pe + n_m_r * self.tau_sl)
                delta_g = (delta_tau + self.tau_sl) * (platoons_g[0].n_pc + n_m_g) + n_m_g * tau_idle_g/2
                if delta_g + delta_r > 0: #check +
                    return platoons_g[0].tau_pe
        return 0

    def performPBS(self, tau_ge, s_r, s_g):
        platoons_r = [c for c in s_r if c.status == 3]
        if len(platoons_r) > 0:
            tau_gremain = self.tau_g_max - tau_ge
            n_q = self.calcN_q(s_r)
            n_m_r = sum([c.n_pc for c in s_r if c.status == 1 and c.tau_ps_t < platoons_r[0].tau_ps_t])
            tau_qa_r = self.getTau_qc(n_q + n_m_r, 0)
            tau_qa_r = max(tau_qa_r, tau_gremain)
            tau_idle_r = platoons_r[0].tau_ps_t - tau_qa_r - self.tau_y
            '''
            tbd
            '''
        return 0

        



    def aggregateClusters(self, s):
        currentClusterInd = 0
        while(currentClusterInd < len(s)-1):
            currentCluster = s[currentClusterInd]
            nextCluster = s[currentClusterInd+1]
            currentTau_pe = currentCluster.tau_pe
            if nextCluster.tau_ps_t - currentTau_pe < self.c_thc:
                #merge tuples
                currentCluster.tau_pe = nextCluster.tau_pe
                currentCluster.tau_pd = currentCluster.tau_pe - currentCluster.tau_ps_t #changed from sum to this
                currentCluster.n_pc += nextCluster.n_pc
                currentCluster.calcQ_pc()
                del s[currentClusterInd+1]
            else:
                currentClusterInd += 1

        for cluster in s:
            if cluster.tau_ps_t <= 0:
                cluster.status = 2
            elif cluster.q_pc > self.c_thpd and cluster.n_pc > self.c_thpc:
                cluster.status = 3
            else:
                cluster.status = 1
        return s

    def estimateN_qa(self, s, tau_ge, tau_adv, n_qn):
        n_qa = n_qn
        tau_qc = self.getTau_qc(n_qa, tau_ge)
        for c in s:
            if c.tau_ps_t - tau_adv <= tau_qc:
                delta_d = 1/self.tau_sh - c.q_pc
                if delta_d <= 0 or c.tau_pe - tau_adv <= tau_qc:
                    n_qa += c.n_pc
                else:
                    delta_t = (tau_qc - (c.tau_ps_t - tau_adv)) * c.q_pc/delta_d
                    if delta_t < c.tau_pd:
                        n_qa += c.n_pc * delta_t/c.tau_pd
                        return n_qa
                    else:
                        n_qa += c.n_pc

    def calcN_q(self, s):
        return sum([c.n_pc for c in s if c.tau_ps_t <= 0])


    def getTau_qc(self, n_q, tau_ge):
        #calculate queue clearing time
        return self.tau_sl - tau_ge + self.tau_sh * n_q if tau_ge < self.tau_sl else self.tau_sh * n_q
            

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
        #print(self.phaseArr)

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

    
class HillClimbing():
    def __init__(self, evalFunc, params, stepSizes):
        self.evalFunc = evalFunc
        self.params = params
        self.stepSizes = stepSizes

        self.posDirection = 1
        self.negDirection = -1
        self.fitness = self.evalFunc(self.params)
        self.best = [self.fitness, self.params]
    
    def optimize(self, epsilon=1, numRuns=1, maxIter=50, strategy=0, paramValidCallbacks=None):
        '''
        strategy: 
            0 = calc all directions, take best and multiply with gradient
            1 = calc all directions, get all fitness increasing gradients and summarize as one gradient --> updates are performed in several directions at once
            2 = iterate over directions and make the step and update for one direction immediately, if a better fitness value is achieved
        '''
        self.start = time.time()
        self.epsilon = epsilon
        self.numRuns = numRuns
        print("NumRuns: ", self.numRuns)
        print("maxIter: ", maxIter)

        self.fitnessDynamics = [self.fitness]

        for i in range(maxIter):
            if strategy == 0:
                gradient = self._calcGradientUpdateOne()
            elif strategy == 1:
                gradient = self._calcGradientUpdateAll()
            elif strategy == 2:
                gradient = self._calcOneUpdateOne()
            print("Iteration %i done." % (i+1))
            print(gradient)
            print(np.linalg.norm(gradient))
            if any(gradient) and np.linalg.norm(gradient) > self.epsilon: #commented in norm(gradient)
                self.params = self.params + gradient / self.stepSizes if strategy != 2 else self.params #changed to /
                if paramValidCallbacks:
                    for callback in paramValidCallbacks:
                        self.params = callback(self.params)
                self.fitness = self._performRuns(self.params) if strategy != 2 else self.fitness
                print(self.fitness)
                self.fitnessDynamics.append(self.fitness)
                if self.fitness > self.best[0]:
                    self.best = [self.fitness, self.params]
            else:
                break
        self.totalSeconds = time.time()-self.start 
        minutes = int(self.totalSeconds / 60)
        seconds = self.totalSeconds - minutes*60
        print("%d min and %f seconds needed." % (minutes, seconds))
        print("Last evaluation:")
        print("Fitness:", self.fitness)
        print("Params:", self.params)
        print()
        print("Best:")
        print("Optimal fitness:", self.best[0])
        print("Optimal params:", self.best[1])
        
        plt.plot(self.fitnessDynamics)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Fitness dynamics")
        plt.show()

    def _calcGradientUpdateOne(self):
        fitnessDevResults = []
        fitnesses = []
        for i in range(len(self.params)):
            direction = np.zeros(len(self.params))
            direction[i] = self.posDirection
            posParams = self.params+direction*self.stepSizes
            posFitness = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness = self._performRuns(negParams)
            fitnessDevResults.append(posFitness - self.fitness) 
            fitnesses.append(posFitness)
            fitnessDevResults.append(negFitness - self.fitness)
            fitnesses.append(negFitness)
        maxInd = np.argmax(fitnessDevResults)
        if maxInd % 2 == 0:
            maxFitnessDev = fitnessDevResults[maxInd]
        else:
            maxFitnessDev = -fitnessDevResults[maxInd]
        gradient = np.zeros(len(self.params))
        gradient[int(np.floor(maxInd/2))] = maxFitnessDev
        return gradient
    
    def _calcGradientUpdateAll(self):
        gradient = []
        fitnesses = []
        for i in range(len(self.params)):
            direction = np.zeros(len(self.params))
            direction[i] = self.posDirection
            posParams = self.params+direction*self.stepSizes
            posFitness = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness = self._performRuns(negParams)
            gradient.append(((posFitness - negFitness)/2)) 
            fitnesses.append(posFitness)
            fitnesses.append(negFitness)
        return np.array(gradient)

    def _calcOneUpdateOne(self):
        fitnesses = []
        gradient = []
        for i in range(len(self.params)):
            direction = np.zeros(len(self.params))
            direction[i] = self.posDirection
            posParams = self.params+direction*self.stepSizes
            posFitness = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness = self._performRuns(negParams)
            if posFitness > self.fitness and posFitness > negFitness:
                self.params = posParams
                self.fitness = posFitness
                gradient.append(1)
            elif negFitness > self.fitness and negFitness > posFitness:
                self.params = negParams
                self.fitness = negFitness
                gradient.append(-1)
            else:
                gradient.append(0)
            fitnesses.append(posFitness)
            fitnesses.append(negFitness)
        return np.array(gradient)

    def _performRuns(self, params):
        fitnesses = []
        #@ray.remote
        def _runRuns(params):
            return self.evalFunc(params)

        #fitnesses = ray.get([_runRuns.remote(params) for _ in range(self.numRuns)])
        fitnesses = [_runRuns(params) for _ in range(self.numRuns)]
        return np.mean(fitnesses)

'''
Validation callback functions
'''
def checkCTFactor(params):
    ctFactor = params[0]
    if ctFactor < 0.75:
        params[0] = 0.75
    elif ctFactor > 1.5:
        params[0] = 1.5
    return params

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

def meanSpeedAdaSOTL(params):
    def _run(adaSotls):
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for sotl in adaSotls:
                sotl.step()
            step += 1
        traci.close()
        sys.stdout.flush()

    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("2x3.sumocfg")
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

    setFlows(numVehicles, simulationTime)
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])
    
    _run(adaSotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed)

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
    configPath = os.path.abspath("2x3.sumocfg")
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
    os.system('jtrrouter -c 2x3.jtrrcfg')

    traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])
    
    _run(sotls)

    meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
    return float(meanSpeed)

'''
Helper functions
'''

def getTLPhaseInfo():
    tree = ET.parse("2x3net.net.xml")
    root = tree.getroot()
    tls = root.find('tlLogic')
    numPhases = len(tls)
    yellowPhaseDurations = tls[1].attrib['duration']
    return numPhases, int(yellowPhaseDurations)

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