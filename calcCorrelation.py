import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sb

def calcCorr():
    allMeanSpeeds = []
    allStdSpeeds = []
    allMeanWaitingTimes = []
    allStdWaitingTimes = []
    parentDir = "./arterial/Plots/"
    dirs = [dir for dir in os.listdir(parentDir) if os.path.isdir(parentDir+dir)]
    for dir in dirs:
        with open(parentDir+dir+"/dynamicsData.pickle", "rb") as f:
            data = pickle.load(f)
        allMeanSpeeds += data['meanSpeed'][1:]
        allMeanWaitingTimes += data['meanWaitingTime']
        allStdSpeeds += data['stdMeanSpeed']
        allStdWaitingTimes += data['stdMeanWaitingTime']
    frame = pd.DataFrame()
    frame['meanSpeed'] = allMeanSpeeds
    frame['meanWaitingTime'] = allMeanWaitingTimes
    frame['stdSpeed'] = allStdSpeeds
    frame['stdWaitingTimes'] = allStdWaitingTimes
    corr = frame.corr()
    print(corr) #-0.910018 for meanSpeed and meanWaitingTime
    sb.heatmap(corr)
    #plt.savefig(fname="corrMat")
    plt.show()

def createPlotMeanSpeed():
    meanSpeedOptPath = "./arterial/Plots/HillClimbing_AdaSOTL_arterial_5runs_strat1_1200veh_rt2_16/dynamicsData.pickle"
    meanWaitOptPath = "./arterial/Plots/MeanWaitingTime_HillClimbing_AdaSOTL_arterial_5runs_strat1_1200veh_rt2_16/dynamicsData.pickle"
    with open(meanSpeedOptPath, "rb") as f:
        meanSpeedData = pickle.load(f)
    with open(meanWaitOptPath, "rb") as f:
        meanWaitingData = pickle.load(f)
    plt.plot(meanSpeedData['meanSpeed'], "r", label="Mean speed optimized")
    plt.plot(meanWaitingData['meanSpeed'], label="Mean waiting time optimized")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Speed")
    plt.legend(loc="lower right")
    plt.title("Mean speed dynamics comparison")
    plt.show()

def createPlotMeanWaitingTime():
    meanSpeedOptPath = "./arterial/Plots/HillClimbing_AdaSOTL_arterial_5runs_strat1_1200veh_rt2_16/dynamicsData.pickle"
    meanWaitOptPath = "./arterial/Plots/MeanWaitingTime_HillClimbing_AdaSOTL_arterial_5runs_strat1_1200veh_rt2_16/dynamicsData.pickle"
    with open(meanSpeedOptPath, "rb") as f:
        meanSpeedData = pickle.load(f)
    with open(meanWaitOptPath, "rb") as f:
        meanWaitingData = pickle.load(f)
    plt.plot(meanSpeedData['meanWaitingTime'], "r", label="Mean speed optimized")
    plt.plot(meanWaitingData['meanWaitingTime'], label="Mean waiting time optimized")
    plt.xlabel("Iteration")
    plt.ylabel("Mean waiting time")
    plt.legend(loc="upper right")
    plt.title("Mean waiting time dynamics comparison")
    plt.show()

if __name__ == '__main__':
    #calcCorr()
    createPlotMeanWaitingTime()