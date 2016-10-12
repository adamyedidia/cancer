import openpyxl
import numpy as np
import string
import random
import matplotlib.pyplot as p
from sklearn import linear_model

wb = openpyxl.load_workbook('../spreadsheets/data_pruned_h_bin_no_dummy_no_density.xlsx')
mainSheet = wb.active

def convertToColumnLabel(x):
    lastDigit = string.ascii_uppercase[x % 26]
        
    dividend = x - (x%26)
    
    if dividend == 0:
        return lastDigit
    
    else:
        return convertToColumnLabel(dividend/26 - 1) + lastDigit        

def turnDataListIntoTwoArrays(dataList):
    x = np.array([i[0] for i in dataList])
    y = np.array([i[1] for i in dataList])
    
    return x,y

def turnDataListIntoArrayAndList(dataList):
    x = np.array([i[0] for i in dataList])
    y = [i[1][0] for i in dataList]
    
    return x,y

def printCoeffsForWholeDataSet(dataList):
    x, y = turnDataListIntoTwoArrays(dataList)

    regr = linear_model.LogisticRegression()
    regr.fit(x, y)

    for j, coef in enumerate(regr.coef_[0]):
        print mainSheet[convertToColumnLabel(j) + "1"].value, coef

def train(regr, trainingSet):
    x, y = turnDataListIntoTwoArrays(trainingSet)
    regr.fit(x, y)

def test(regr, testSet, threshold):
    x, y = turnDataListIntoArrayAndList(testSet)
        
    predictions = [i[0] for i in regr.predict_proba(x)]
    
    ourAnswers = [i > threshold for i in predictions]
    
    falsePositives = [i and (not j) for i, j in zip(ourAnswers, y)]
    falseNegatives = [(not i) and j for i, j in zip(ourAnswers, y)]
    
    
    print "Out of", len(ourAnswers), "data points, there were", \
        sum(falsePositives), "false positives, and", sum(falseNegatives), \
        "false negatives."

    return sum(falsePositives), sum(falseNegatives)

MAX_ROW = 1142
ANSWER_COLUMN = "CL"

dataList = []

for i in range(2, MAX_ROW+1):
    newDataPoint = [[]]
    j = 0
    currentColumnName = "A"
    
    while not currentColumnName == ANSWER_COLUMN:
        newDataPoint[0].append(float(mainSheet[currentColumnName + str(i)].value))
        
        j += 1
        currentColumnName = convertToColumnLabel(j)
        
    newDataPoint.append([float(mainSheet[ANSWER_COLUMN + str(i)].value)])
        
    dataList.append(newDataPoint)        
    
TRAINING_SET_FRACTION = 0.8    
    
trainingSetSize = int(TRAINING_SET_FRACTION*len(dataList)) 

random.shuffle(dataList)

trainingSet = dataList[:trainingSetSize]
testSet = dataList[trainingSetSize:]

regr = linear_model.LogisticRegression()

train(regr, trainingSet)

FALSE_POSITIVE_PENALTY = 1
FALSE_NEGATIVE_PENALTY = 10

falsePositiveList = []
falseNegativeList = []
thresholdList = []

for thresholdTimesAHundred in range(-100, 200):
#for thresholdTimesAHundred in range(50, 60):
    
    threshold = thresholdTimesAHundred/100.
    
#    random.shuffle(dataList)

#    trainingSet = dataList[:trainingSetSize]
#    testSet = dataList[trainingSetSize:]

#    regr = linear_model.LogisticRegression()

#    train(regr, trainingSet)
    
    falsePositives, falseNegatives = test(regr, testSet, threshold)
    
    falsePositiveList.append(falsePositives)
    falseNegativeList.append(falseNegatives)
    thresholdList.append(threshold)
    
p.plot(thresholdList, falsePositiveList, "b-")
p.plot(thresholdList, falseNegativeList, "r-")

p.savefig("false_pos_neg.png")

p.clf()

penalties = [FALSE_POSITIVE_PENALTY*i + FALSE_NEGATIVE_PENALTY*j \
    for i,j in zip(falsePositiveList, falseNegativeList)]
    
p.plot(thresholdList, penalties, "b-")

p.savefig("penalties.png")

#print vectorizer

#print wb["all"]["A1"].value