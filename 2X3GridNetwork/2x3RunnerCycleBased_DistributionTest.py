#!/usr/bin/env python

from numpy.core.fromnumeric import mean
from helperTwoPhases import *
import matplotlib.pyplot as plt

def run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths):
    step = 0
    pathCounter = 0
    cycleBasedTLControllers = []
    while traci.simulation.getMinExpectedNumber() > 0:
        if step % 1200 == 0 and step < 3600:
            mapLPDetailsToTL(trafficLights, lpSolveResultPaths[pathCounter])
            maxNodeUtilization = max([tl.utilization for tl in trafficLights])
            cycleTime = int(np.round(ctFactor * ((1.5 * 2*3 + 5)/(1 - maxNodeUtilization)))) #maybe edit hard coded yellow phases and extract them from file
            #print(cycleTime)
            pathCounter += 1
            for counter, tl in enumerate(trafficLights):
                cycleBasedTLControllers.append(CycleBasedTLController(tl, cycleTime, phaseShifts[counter]))
        
        for controller in cycleBasedTLControllers:
            controller.step()
        traci.simulationStep()

        step += 1
    traci.close()
    sys.stdout.flush()

if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    sumoGui = checkBinary('sumo-gui')
    configPath = os.path.abspath("2x3.sumocfg")
    simulationTime = 3600
    numVehicles = 900
    ctFactor = 0.9
    #phaseShifts = [0]*6 #6 for 6 junctions 
    phaseShifts = [0, 10, 20, 10, 20, 30]
    lpSolveResultPaths = ['./LPSolve/2x3Grid_a_eps0,4.lp.csv', './LPSolve/2x3Grid_b_eps0,4.lp.csv', './LPSolve/2x3Grid_c_eps0,4.lp.csv']

    meanSpeeds = []
    meanWaitingTimes = []
    numReplications = 30

    for i in range(numReplications):

        #create instances
        trafficLights = createTrafficLights()

        setFlows(numVehicles, simulationTime)
        os.system('jtrrouter -c 2x3.jtrrcfg')

        traci.start([sumoBinary, "-c", configPath,
                                        "--tripinfo-output", "tripinfo.xml",
                                        "--statistic-output", "statistics.xml"])

        run(trafficLights, ctFactor, phaseShifts, lpSolveResultPaths)

        meanSpeed, meanWaitingTime = getMeanSpeedWaitingTime()
        meanSpeeds.append(float(meanSpeed))
        meanWaitingTimes.append(float(meanWaitingTime))
        print("Replication %i done." % (i+1))

    sortedSpeeds = sorted(meanSpeeds)
    sortedWaitingTimes = sorted(meanWaitingTimes)
    plt.plot(sortedSpeeds)
    plt.title("Mean speeds")
    plt.show()
    plt.plot(sortedWaitingTimes)    
    plt.title("Mean Waiting times")
    plt.show()