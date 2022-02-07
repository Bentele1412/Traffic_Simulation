import pickle
import numpy as np

if __name__ == '__main__':
    pathToFile = "./GridNetwork/Plots/HillClimbing_AdaSOTL_2x3_fixed_5runs_strat1_1500veh/dynamicsData.pickle"
    with open(pathToFile, "rb") as f:
        data = pickle.load(f)
    maxSpeedInd = np.argmax(data['meanSpeed'])
    print(data['meanWaitingTime'][maxSpeedInd])