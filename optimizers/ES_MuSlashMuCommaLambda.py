#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool

class ES_MuSlashMuCommaLambda():
    def __init__(self, parent, mu, lambda_):
        self.parent = parent
        self.mu = mu
        self.lambda_ = lambda_
        self.tau = 1/np.sqrt(2*len(parent))
    
    def optimize(self, evalFunc, plotFolderPath, sigma=1, sigma_stop=1e-5, isMaximization=False, numRuns=1, maxIter=50):
        self.start = time.time()

        self.sigma_stop = sigma_stop
        self.isMaximization = isMaximization
        self.evalFunc = evalFunc
        self.plotFolderPath = plotFolderPath
        self.maxIter = maxIter
        self.numRuns = numRuns

        self.sigma = sigma
        self.rng = np.random.default_rng(12345)
        self.fitnessDynamics = []
        self.stdMeanSpeeds = []
        self.meanWaitingTimes = []
        self.stdWaitingTimes = []
        self.sigmaDynamics = []
        g = 0

        while self.sigma > self.sigma_stop:
            offsprings = []
            for _ in range(self.lambda_):
                sigma_l = self.sigma*np.e**(self.tau*self.rng.normal())
                s_l = self.rng.normal(size=len(self.parent))
                offspring_l = np.add(self.parent, sigma_l*s_l)
                #print(offspring_l)
                fitness_l, _, _, _ = self._performRuns(offspring_l)
                offsprings.append([fitness_l, offspring_l, sigma_l])
            if self.isMaximization:
                sortedOffsprings = sorted(offsprings, key=lambda x: x[0], reverse=True)
            else:
                sortedOffsprings = sorted(offsprings, key=lambda x: x[0])
            if self.mu != 1:
                fitnesses = 0
                offsprings = [0]*len(self.parent)
                sigma = 0
                for f, o, s in sortedOffsprings[:self.mu]:
                    fitnesses += f
                    offsprings = np.add(offsprings, o)
                    sigma += s
                recombination = [(1/self.mu)*fitnesses, (1/self.mu)*offsprings, (1/self.mu)*sigma]
            else:
                recombination = sortedOffsprings[0]
            self.parent = recombination[1]
            self.sigma = recombination[-1]
            fitness, stdMeanSpeed, meanWaitingTime, stdWaitingTime = self._performRuns(self.parent)
            self.fitnessDynamics.append(fitness)
            self.sigmaDynamics.append(self.sigma)
            self.stdMeanSpeeds.append(stdMeanSpeed)
            self.meanWaitingTimes.append(meanWaitingTime)
            self.stdWaitingTimes.append(stdWaitingTime)

            if g == 0:
                self.best = [self.parent, fitness]
            else:
                if isMaximization:
                    if self.fitness > self.best[-1]:
                        self.best = [self.parent, fitness]
                else:
                    if self.fitness < self.best[-1]:
                        self.best = [self.parent, fitness]
            

            g += 1
            print("Generation %i done." % g)
            if g == self.maxIter:
                break
        self.totalSeconds = time.time()-self.start 
        minutes = int(self.totalSeconds / 60)
        seconds = self.totalSeconds - minutes*60
        
        #Optimization results
        print("%d min and %f seconds needed." % (minutes, seconds))
        print("Last evaluation:")
        print("Optimal fitness:", self.fitnessDynamics[-1])
        print("Optimal params:", self.parent)
        print()
        print("Best:")
        print("Optimal fitness:", self.best[0])
        print("Optimal params:", self.best[-1])
        
        #Dynamics ploting
        plt.plot(self.fitnessDynamics)
        plt.xlabel("Generation")
        plt.ylabel("Mean Speed")
        plt.title("Mean Speed dynamics")
        plt.savefig(self.plotFolderPath + "meanSpeed.png")
        plt.show()

        plt.plot(self.sigmaDynamics)
        plt.xlabel("Generation")
        plt.ylabel("Sigma")
        plt.title("Sigma dynamics")
        plt.savefig(self.plotFolderPath + "sigma.png")
        plt.show()

        plt.plot(self.stdMeanSpeeds)
        plt.xlabel("Generation")
        plt.ylabel("Std Mean Speed")
        plt.title("Std Mean Speed dynamics")
        plt.savefig(self.plotFolderPath + "stdMeanSpeed.png")
        plt.show()

        plt.plot(self.meanWaitingTimes)
        plt.xlabel("Generation")
        plt.ylabel("Mean waiting time")
        plt.title("Mean waiting time dynamics")
        plt.savefig(self.plotFolderPath + "meanWaitingTime.png")
        plt.show()

        plt.plot(self.stdWaitingTimes)
        plt.xlabel("Generation")
        plt.ylabel("Std waiting time")
        plt.title("Std waiting time dynamics")
        plt.savefig(self.plotFolderPath + "stdMeanWaitingTime.png")
        plt.show()

        #save dynamics
        dynamics = {'meanSpeed': self.fitnessDynamics,
                'sigmaDynamics': self.sigmaDynamics,
                'stdMeanSpeed': self.stdMeanSpeeds,
                'meanWaitingTime': self.meanWaitingTimes,
                'stdMeanWaitingTime': self.stdWaitingTimes}
        with open(self.plotFolderPath + "dynamicsData.pickle", 'wb') as f:
            pickle.dump(dynamics, f)

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