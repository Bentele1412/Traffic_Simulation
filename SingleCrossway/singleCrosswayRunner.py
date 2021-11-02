#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
from sumolib import checkBinary

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def run():
    step = 0
    traci.trafficlight.setPhase("m", 0)
    detectedVehicles = 0
    lastVehicleIDs = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #detectedVehicles += traci.inductionloop.getLastStepVehicleNumber("detect")
        vehicleIDs = traci.inductionloop.getLastStepVehicleIDs("detect")
        for id in vehicleIDs:
            if not id in lastVehicleIDs:
                detectedVehicles += 1
        lastVehicleIDs = vehicleIDs
        step += 1
    print("Vehicles passed detector: ", detectedVehicles)
    traci.close()
    sys.stdout.flush()


if __name__ == '__main__':
    sumoBinary = checkBinary('sumo-gui')
    configPath = os.path.abspath("circularControl.sumocfg")
    netPath = os.path.abspath("circularControlNet.net.xml")

    traci.start([sumoBinary, "-c", configPath,
                             "--tripinfo-output", "tripinfo.xml",
                             "--statistic-output", "statistics.xml"])

    run()