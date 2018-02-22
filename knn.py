import pandas as pd
import numpy as np
import sys
import math

def to_numeric(df):
    last = df.shape[1]-1
    cols = list(range(0, last))
    df[cols] = df[cols].apply(pd.to_numeric)
    return df

def distance(a,b):
    d = 0
    for i in range(len(a)):
        d += math.pow((a[i]-b[i]), 2)
    distance = math.sqrt(d)
    return distance

def knn(testset, trainingset, k):
    testset[testset.shape[1]-1] = np.NaN
    for i, row in testset.iterrows():
        neighbors_ = neighbors(row, trainingset, k)
        label = majority(neighbors_)
        testset.loc[i,testset.shape[1]-1] = label
    return testset



def neighbors(instance, trainingset, k):
    label_index = trainingset.shape[1]-1
    distances = []
    for i, r in trainingset.iterrows():
        d = distance(list(instance)[0:label_index-1], list(r)[0:label_index-1])
        distances.append((list(r)[label_index], d))
    distances.sort(key=lambda x: x[1])
    k_neighbors = []
    for j in range(k):
        k_neighbors.append(distances[j][0])
    return k_neighbors

def majority(neighbors_):
    return max(neighbors_, key=neighbors_.count)


def main():
    # read in csv
    if len(sys.argv) > 3:
        k = int(sys.argv[1])
        for file in sys.argv[2:]:
            training_data = pd.read_csv(sys.argv[2], header=None)
            testing_data = pd.read_csv(sys.argv[3], header=None)
            testing_data = to_numeric(testing_data)
            training_data = to_numeric(training_data)
            # print(testing_data, 'String')
            testing_data = knn(testing_data, training_data, k)
            testing_data.to_csv("testing_k"+str(k), header=False, index=False)
    else:
        print("Please enter a file name.")


main()
