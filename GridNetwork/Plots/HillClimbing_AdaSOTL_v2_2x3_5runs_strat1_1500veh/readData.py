import pickle

with open("dynamicsData.pickle", 'rb') as f:
    dynamics = pickle.load(f)
    print(dynamics)