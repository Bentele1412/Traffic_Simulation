#!/usr/bin/env python
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

def getMeanSpeed():
    tree = ET.parse("statistics.xml")
    root = tree.getroot()
    return root.find('vehicleTripStatistics').attrib['speed']

def changeDuration(duration):
    tree = ET.parse("twoCross.net.xml")
    root = tree.getroot()
    for counter, phase in enumerate(root.iter("phase")):
        if counter % 2 == 0:
            phase.set("duration", str(duration))
    tree.write("twoCross.net.xml")

def run():
    step = 0
    traci.trafficlight.setPhase("m", 0)
    detectedVehicles = 0
    lastVehicleIDs = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        '''
        #detectedVehicles += traci.inductionloop.getLastStepVehicleNumber("detect")
        vehicleIDs = traci.inductionloop.getLastStepVehicleIDs("detect")
        for id in vehicleIDs:
            if not id in lastVehicleIDs:
                detectedVehicles += 1
        lastVehicleIDs = vehicleIDs
        '''
        step += 1
    #print("Vehicles passed detector: ", detectedVehicles)
    traci.close()
    sys.stdout.flush()


if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("twoCross.sumocfg")
    netPath = os.path.abspath("twoCross.net.xml")

    #optimization loop
    greenDuration = 0
    direction = 1
    #changeDuration(greenDuration)
    maxGreenDuration = 2

    meanSpeeds = np.zeros(maxGreenDuration, dtype=float)

    numReplications = 1

    for i in range(1, maxGreenDuration+1):
        greenDuration += direction
        #changeDuration(greenDuration)
        print("Green duration: ", greenDuration)
        for rep in range(numReplications):
            
            #create new routes
            os.system('randomTrips.py -n twoCross.net.xml -o ".\\twoCrossFlow.xml" -b 0 -e 3600 --random --binomial 1 -p 6 --trip-attributes="departPos=\\"random\\""')
            os.system('jtrrouter -c twoCross.jtrrcfg')

            #start server
            traci.start([sumoBinary, "-c", configPath,
                                    "--tripinfo-output", "tripinfo.xml",
                                    "--statistic-output", "statistics.xml"])

            #one simulation run
            run()

            meanSpeeds[i-1] += float(getMeanSpeed())
            print(getMeanSpeed())
        meanSpeeds[i-1] /= numReplications
    
    greenDurations = np.arange(1, maxGreenDuration+1, 1)
    plt.plot(greenDurations, meanSpeeds)
    plt.yticks(np.arange(1, 10, 1))
    plt.title("Mean speeds of various green phase durations")
    plt.xlabel("Green phase duration")
    plt.ylabel("Mean speed")
    plt.show()
