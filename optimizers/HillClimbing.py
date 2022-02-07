#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ray
from multiprocessing import Pool


class HillClimbing():
    def __init__(self, evalFunc, params, stepSizes):
        self.evalFunc = evalFunc
        self.params = params
        self.stepSizes = stepSizes

        self.posDirection = 1
        self.negDirection = -1
        self.fitness, _ = self.evalFunc(self.params)
        self.best = [self.fitness, self.params]

        self.meanWaitingTimes = []
        self.stdWaitingTimes = []
        self.stdMeanSpeeds = []
        self.paramsDynamics = []
        self.gradientDynamics = []
    
    def optimize(self, plotFolderPath, epsilon=1, numRuns=1, maxIter=50, strategy=0, paramValidCallbacks=None, useGradient=True, reduceStepSize=False):
        '''
        strategy: 
            0 = calc all directions, take best and multiply with gradient
            1 = calc all directions, get all fitness increasing gradients and summarize as one gradient --> updates are performed in several directions at once
            2 = iterate over directions and make the step and update for one direction immediately, if a better fitness value is achieved
        '''
        self.start = time.time()
        #ray.init()
        self.epsilon = epsilon
        self.numRuns = numRuns
        self.plotFolderPath = plotFolderPath
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
                if useGradient:
                    self.params = self.params + gradient * self.stepSizes if strategy != 2 else self.params
                else:
                    goodDirections = np.array(list(map(lambda x: np.sign(x), gradient)))
                    self.params = self.params + goodDirections * self.stepSizes if strategy != 2 else self.params
                if paramValidCallbacks:
                    for callback in paramValidCallbacks:
                        self.params = callback(self.params)
                self.fitness, stdMeanSpeed, meanWaitingTime, stdMeanWaitingTime = self._performRuns(self.params) # (commented out) if strategy != 2 else self.fitness
                print(self.fitness)
                self.fitnessDynamics.append(self.fitness)
                self.meanWaitingTimes.append(meanWaitingTime)
                self.stdMeanSpeeds.append(stdMeanSpeed)
                self.stdWaitingTimes.append(stdMeanWaitingTime)
                self.paramsDynamics.append(self.params)
                self.gradientDynamics.append(gradient)
                if self.fitness > self.best[0]:
                    self.best = [self.fitness, self.params]
            elif reduceStepSize:
                self.stepSizes = np.array(self.stepSizes)/2
            else:
                break
        self.totalSeconds = time.time()-self.start 
        minutes = int(self.totalSeconds / 60)
        seconds = self.totalSeconds - minutes*60
        
        #Optimization results
        print("%d min and %f seconds needed." % (minutes, seconds))
        print("Last evaluation:")
        print("Fitness:", self.fitness)
        print("Params:", self.params)
        print()
        print("Best:")
        print("Optimal fitness:", self.best[0])
        print("Optimal params:", self.best[1])
        
        #Dynamics ploting
        plt.plot(self.fitnessDynamics)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Speed")
        plt.title("Mean Speed dynamics")
        plt.savefig(self.plotFolderPath + "meanSpeed.png")
        plt.show()

        plt.plot(self.stdMeanSpeeds)
        plt.xlabel("Iteration")
        plt.ylabel("Std Mean Speed")
        plt.title("Std Mean Speed dynamics")
        plt.savefig(self.plotFolderPath + "stdMeanSpeed.png")
        plt.show()

        plt.plot(self.meanWaitingTimes)
        plt.xlabel("Iteration")
        plt.ylabel("Mean waiting time")
        plt.title("Mean waiting time dynamics")
        plt.savefig(self.plotFolderPath + "meanWaitingTime.png")
        plt.show()

        plt.plot(self.stdWaitingTimes)
        plt.xlabel("Iteration")
        plt.ylabel("Std waiting time")
        plt.title("Std waiting time dynamics")
        plt.savefig(self.plotFolderPath + "stdMeanWaitingTime.png")
        plt.show()

        #save dynamics data
        dynamics = {'meanSpeed': self.fitnessDynamics,
                'stdMeanSpeed': self.stdMeanSpeeds,
                'meanWaitingTime': self.meanWaitingTimes,
                'stdMeanWaitingTime': self.stdWaitingTimes,
                'gradientDynamics': self.gradientDynamics,
                'paramsDynamics': self.paramsDynamics}
        with open(self.plotFolderPath + "dynamicsData.pickle", 'wb') as f:
            pickle.dump(dynamics, f)

    def _calcGradientUpdateOne(self):
        fitnessDevResults = []
        fitnesses = []
        for i in range(len(self.params)):
            direction = np.zeros(len(self.params))
            direction[i] = self.posDirection
            posParams = self.params+direction*self.stepSizes
            posFitness, _, _, _ = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness, _, _, _ = self._performRuns(negParams)
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
            posFitness, _, _, _ = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness, _, _, _ = self._performRuns(negParams)
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
            posFitness, _, _, _ = self._performRuns(posParams)
            direction[i] = self.negDirection
            negParams = self.params+direction*self.stepSizes
            negFitness, _, _, _ = self._performRuns(negParams)
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
        waitingTimes = []
    
        p = Pool()
        results = p.map(self._runRuns, [params]*self.numRuns)
        #results = ray.get([_runRuns.remote(x) for x in range(self.numRuns)])
        for meanSpeed, meanWaitingTime in results:
            #meanSpeed, meanWaitingTime = _runRuns(params)
            fitnesses.append(meanSpeed)
            waitingTimes.append(meanWaitingTime)
        #fitnesses = [_runRuns(params) for _ in range(self.numRuns)]
        return np.mean(fitnesses), np.std(fitnesses), np.mean(waitingTimes), np.std(waitingTimes)

    #@ray.remote
    def _runRuns(self, params):
        return self.evalFunc(params)