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

if __name__ == '__main__':
    calcCorr()