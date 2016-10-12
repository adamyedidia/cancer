import openpyxl, pprint
import numpy as np
import string
import sys
import numbers
import re
import random
from sklearn import linear_model
from sklearn.svm import SVC
import matplotlib.pyplot as p

MAX_ROW = 1779

LIST_OF_BINARY_COLS = ["RACE","PBC","SMOKE","ALCOHOL",
    "ASHKENAZI","HORMONE","ASSESS","DENSITY","FIRSTMAM","FROMSCREEN", "HORMONE",
    "FINDING","FINDTYPE","BXTYPE","BXPATH_CONCAT"]

LIST_OF_REAL_COLS = ["AGE","FBC_1","FBC_M","FBC_NO","POC_COUNT","WEIGHT",
    "HEIGHT","MALE","BIRTH","AGEPREG","MAAGE","MPAGE","PRIORBX"]

LIST_OF_EXCLUDED_COLS = ["ID","PACC","SDATE","STUDY","LOC","SPROCS","BXRESULT",
    "BXDATE", "BXPATHREP"]

SHEET_NAME = "nice_sheet_aug_1.xlsx"


def strSum(s):
    result = ""
    for c in s:
        result += c
    return result

def f1Score(truePositives, trueNegatives, falsePositives, falseNegatives):

    if truePositives == 0:
        return 0

    precision = float(truePositives) / (falsePositives + truePositives)
    recall = float(truePositives) / (truePositives + falseNegatives)

    return 2*(precision*recall)/(precision+recall)

def turnDataListIntoArrayAndList(dataList):
    x = np.array([i[0] for i in dataList])
    y = [i[1][0] for i in dataList]

    return x,y

def convertToColumnLetter(x):
    lastDigit = string.ascii_uppercase[x % 26]

    dividend = x - (x%26)

    if dividend == 0:
        return lastDigit

    else:
        return convertToColumnLetter(dividend/26 - 1) + lastDigit

def query(row, column, inSheet):
    entry = inSheet[column + str(row)].value

    if type(entry) == type("hello") or type(entry) == type(u"hello"):
        return string.strip(inSheet[column + str(row)].value)

    return inSheet[column + str(row)].value

def getColumnLabel(column, inSheet):
    return query(1, column, inSheet)

def extractLabelToLetterDict(inSheet):
    labelToLetterDict = {}

    columnNum = 0
    columnLetter = "A"
    columnLabel = getColumnLabel(columnLetter, inSheet)

    labelToLetterDict[columnLabel] = columnLetter

    while columnLabel != "SXRESULT":
        columnNum += 1
        columnLetter = convertToColumnLetter(columnNum)
        columnLabel = getColumnLabel(columnLetter, inSheet)

        labelToLetterDict[columnLabel] = columnLetter

    return labelToLetterDict

def extractFeatureDict(labelToLetterDict, inSheet):
    featureDict = {}

    for columnLabel in LIST_OF_BINARY_COLS:

        columnLetter = labelToLetterDict[columnLabel]

        featureDict[columnLabel] = {}

        for rowNum in range(2, MAX_ROW+1):
            cellContents = query(rowNum, columnLetter, inSheet)

            if cellContents != None:
                if type(cellContents) == type("hello") or type(cellContents) == type(u"hello"):
                    cellContentsSplit = string.split(cellContents, ",")

                    for entry in cellContentsSplit:
                        featureDict[columnLabel][str(entry)] = None

                else:
                    featureDict[columnLabel][str(cellContents)] = None

    return featureDict

def extractAveragesDict(labelToLetterDict, inSheet):
    averagesDict = {}

    for columnLabel in LIST_OF_REAL_COLS:
        columnLetter = labelToLetterDict[columnLabel]

        totalValue = 0
        numPoints = 0

        for rowNum in range(2, MAX_ROW):
            cellContents = query(rowNum, columnLetter, inSheet)

            if cellContents != None:
                try:
                    totalValue += cellContents
                    numPoints += 1.

                except:
                    print "Suspicious value", cellContents
                    print "Value ignored"


        averagesDict[columnLabel] = totalValue / numPoints

    return averagesDict

def extractFeatureNamesList(featureDict, averagesDict, inSheet):
    featureNamesList = []
    featureNamesDict = {}

    rowNum = 1

    columnNum = 0
    columnLetter = "A"
    columnLabel = getColumnLabel(columnLetter, inSheet)

    while columnLabel != "SXRESULT":
        if columnLabel in LIST_OF_BINARY_COLS:
            for feature in featureDict[columnLabel]:
                featureName = columnLabel + "_" + str(feature)

                featureNamesList.append(featureName)
                featureNamesDict[featureName] = [0, 0]

        elif columnLabel in LIST_OF_REAL_COLS:
            featureNamesList.append(columnLabel)

        columnNum += 1
        columnLetter = convertToColumnLetter(columnNum)
        columnLabel = getColumnLabel(columnLetter, inSheet)

    return featureNamesList, featureNamesDict

def extractDataList(featureDict, averagesDict, featureNamesDict, inSheet):
    dataList = []

    lastDataPointLength = -1

    numCancer = 0


    for rowNum in range(2,MAX_ROW+1):

        binaryFeaturesOn = []

        columnNum = 0
        columnLetter = "A"
        columnLabel = getColumnLabel(columnLetter, inSheet)

        dataPoint = [[]]

        while columnLabel != "SXRESULT":

            entry = query(rowNum, columnLetter, inSheet)

            if columnLabel in LIST_OF_BINARY_COLS:
                if entry != None:
                    if type(entry) == type("hello") or type(entry) == type(u"hello"):
                        entrySplit = [string.strip(i) for i in string.split(entry, ",")]

                        for feature in featureDict[columnLabel]:
                            if feature in entrySplit:
                                dataPoint[0].append(1)
                                binaryFeaturesOn.append(columnLabel + "_" + feature)
                            else:
                                dataPoint[0].append(0)
                    else:
                        for feature in featureDict[columnLabel]:
                            if str(entry) == feature:
                                dataPoint[0].append(1)
                                binaryFeaturesOn.append(columnLabel + "_" + feature)
                            else:
                                dataPoint[0].append(0)

                else:
                    for feature in featureDict[columnLabel]:
                        dataPoint[0].append(0)

            elif columnLabel in LIST_OF_REAL_COLS:
                if entry != None and (type(entry) == type(1) or type(entry) == type(1.) or type(entry) == type(1L)):
                    dataPoint[0].append(entry)
                else:
                    dataPoint[0].append(averagesDict[columnLabel])

            elif columnLabel in LIST_OF_EXCLUDED_COLS:
                pass

            else:
                raise RuntimeError("Unrecognized column " + columnLabel)

            columnNum += 1
            columnLetter = convertToColumnLetter(columnNum)
            columnLabel = getColumnLabel(columnLetter, inSheet)

        dataPointLength = len(dataPoint[0])

        try:
            assert dataPointLength == lastDataPointLength or lastDataPointLength == -1
        except:
            raise RuntimeError("Bad row length at row " + str(rowNum) + ": " + \
                str(lastDataPointLength) + ", " + str(dataPointLength))

        lastDataPointLength = dataPointLength

        sxresult = query(rowNum, columnLetter, inSheet)

        if sxresult == "M":
            dataPoint.append([1])
            dataList.append(dataPoint)

            for binaryFeature in binaryFeaturesOn:
                featureNamesDict[binaryFeature][0] += 1

            numCancer += 1

        elif sxresult == "H" or sxresult == "B":
            dataPoint.append([0])
            dataList.append(dataPoint)

            for binaryFeature in binaryFeaturesOn:
                featureNamesDict[binaryFeature][1] += 1

    print numCancer

    return dataList

def binarifyColumn(wb, columnName):
    differentValueDict = getDifferentValueDict(wb, columnName)

    newSheet = wb.create_sheet()
    newSheet.title = columnDescriptor

    for i in range(2, MAX_ROW+1):
        for j in range(len(differentValueDict)):
            newSheet[convertToColumnLabel(j) + str(i)].value = 0

    for j, v in enumerate(differentValueDict.keys()):
        newSheet[convertToColumnLabel(j) + "1"].value = columnDescriptor + "_" + v

        for i in differentValueDict[v]:
            newSheet[convertToColumnLabel(j) + str(i)].value = 1

def train(regr, trainingSet):
    x, y = turnDataListIntoTwoArrays(trainingSet)
    regr.fit(x, y)

def printCoeffsForRegression(regr, featureList, featureNamesDict):
    listOfCoeffs = []

    for j, coef in enumerate(regr.coef_[0]):
        listOfCoeffs.append((featureList[j], coef))
        listOfCoeffs.sort(key=lambda x: abs(x[1]))
        listOfCoeffs.reverse()

    output = open("coeffs.txt", "w")

    for t in listOfCoeffs:
        if t[0] in featureNamesDict:
            output.write(str(t[0]) + " " + str(t[1]) + " " + \
                str(featureNamesDict[t[0]][0]) + " " + str(featureNamesDict[t[0]][1]) + "\n")

        else:
            output.write(str(t[0]) + " " + str(t[1]) + "\n")

def turnDataListIntoTwoArrays(dataList):

    x = np.array([i[0] for i in dataList])
    y = np.array([i[1] for i in dataList])

    return x,y

def findOptimalThresholds(regr, validationSet):
    x, y = turnDataListIntoArrayAndList(validationSet)

    #predictions = [i[0] for i in regr.predict(x)]
    predictions = regr.predict(x)

    truePositiveList = []
    trueNegativeList = []
    falsePositiveList = []
    falseNegativeList = []
    thresholdList = []

    for thresholdTimesAHundred in range(-100, 200):

        threshold = thresholdTimesAHundred/100.

        ourAnswers = [i > threshold for i in predictions]

        truePositives = sum([i and j for i, j in zip(ourAnswers, y)])
        trueNegatives = sum([(not i) and (not j) for i, j in zip(ourAnswers, y)])
        falsePositives = sum([i and (not j) for i, j in zip(ourAnswers, y)])
        falseNegatives = sum([(not i) and j for i, j in zip(ourAnswers, y)])

        truePositiveList.append(truePositives)
        trueNegativeList.append(trueNegatives)
        falsePositiveList.append(falsePositives)
        falseNegativeList.append(falseNegatives)
        thresholdList.append(threshold)

    f1Scores = [f1Score(tp, tn, fp, fn) for tp, tn, fp, fn in zip(truePositiveList, \
        trueNegativeList, falsePositiveList, falseNegativeList)]

    bestF1threshold = thresholdList[f1Scores.index(max(f1Scores))]

    falseNegativeList.reverse()
    bestNoCancerThresholdReversed = falseNegativeList.index(0)
    falseNegativeList.reverse()
    bestNoCancerThreshold = thresholdList[len(thresholdList) - bestNoCancerThresholdReversed - 1]

    print f1Scores

    print bestF1threshold, bestNoCancerThreshold
    return bestF1threshold, bestNoCancerThreshold

def test(regr, testSet, threshold):
    x, y = turnDataListIntoArrayAndList(testSet)
#    predictions = [i[0] for i in regr.predict(x)]
    predictions = regr.predict(x)

    ourAnswers = [i > threshold for i in predictions]

    truePositives = sum([i and j for i, j in zip(ourAnswers, y)])
    trueNegatives = sum([(not i) and (not j) for i, j in zip(ourAnswers, y)])
    falsePositives = sum([i and (not j) for i, j in zip(ourAnswers, y)])
    falseNegatives = sum([(not i) and j for i, j in zip(ourAnswers, y)])

    return truePositives, trueNegatives, falsePositives, falseNegatives

def rotatingTraining(dataSet, rotationSize, threshold=0.2, alpha=1.0):

    totalTP = 0
    totalTN = 0
    totalFP = 0
    totalFN = 0

    numPoints = len(dataSet)

    random.shuffle(dataSet)

    i = 0

    while i<numPoints:
        cappedTopRange = min(numPoints, i+rotationSize)
        mainChunkSize = cappedTopRange - i
        miniChunkSize = rotationSize - mainChunkSize
        testSet = dataSet[i:i+mainChunkSize] + dataSet[:miniChunkSize]
        trainingSet = dataSet[miniChunkSize:i] + dataSet[cappedTopRange:]

#        print "Test set starting at point", i

        tp, tn, fp, fn = trainAndTest(trainingSet, testSet, threshold, alpha)

        totalTP += tp
        totalTN += tn
        totalFP += fp
        totalFN += fn

        i += rotationSize

    return totalTP, totalTN, totalFP, totalFN

def trainAndTest(trainingSet, testSet, threshold, alpha):
    regr = linear_model.Ridge(alpha=alpha)

    train(regr, trainingSet)

    tp, tn, fp, fn = test(regr, testSet, threshold)

    return tp, tn, fp, fn

# This approach might cheat?
def find5PercentThresholdRotating(regr, dataSet):
    size = 100
    bestThreshold = -2


    for thresholdWritLarge in range(-1, 15):
        threshold = thresholdWritLarge/float(size)

        tp, tn, fp, fn = rotatingTraining(dataSet, 20, threshold, alpha=30.)

        print threshold, float(fn) / (fn + tp)

        if float(fn) / (fn + tp) <= 0.05:
            bestThreshold = threshold
#            print threshold

    return bestThreshold

def find5PercentThreshold(regr, dataSet):
    size = 100
    bestThreshold = -2


    for thresholdWritLarge in range(-1*size, 2*size):
        threshold = thresholdWritLarge/float(size)

        tp, tn, fp, fn = test(regr, dataSet, threshold)

        print threshold, float(fn) / (fn + tp)

        if float(fn) / (fn + tp) <= 0.05:
            bestThreshold = threshold
#            print threshold

    return bestThreshold


def printStats(tp, tn, fp, fn):
    print "F1 SCORE:", f1Score(tp, tn, fp, fn)
    print "True positives:", tp
    print "True negatives:", tn
    print "False positives:", fp
    print "False negatives:", fn

def mainThreshold():
    alpha = 30.0

    labelToLetterDict = extractLabelToLetterDict(inSheet)
    featureDict = extractFeatureDict(labelToLetterDict, inSheet)
    averagesDict = extractAveragesDict(labelToLetterDict, inSheet)
    featureNamesList, featureNamesDict = extractFeatureNamesList(featureDict, averagesDict, inSheet)

    for featureName in featureNamesList:
        print featureName

    # Warning: modifies featureNamesDict
    dataSet = extractDataList(featureDict, averagesDict, featureNamesDict, inSheet)
    random.shuffle(dataSet)

    trainingSet = dataSet[:int(3.*len(dataSet)/4.)]
    testSet = dataSet[int(3.*len(dataSet)/4.):]

    regr = linear_model.Ridge(alpha=alpha)

    train(regr, trainingSet)
    threshold = find5PercentThresholdRotating(regr, dataSet)

    print "Threshold:", threshold

    tp, tn, fp, fn = rotatingTraining(dataSet, 5, threshold=threshold, alpha=alpha)

    print "Fraction of classified patients", float(tn + fn) / (tp+fp+fn+tn)

    print "Miss rate", float(fn)/float(tp + fn)

    printStats(tp, tn, fp, fn)

    regr = linear_model.Ridge(alpha=alpha)

    train(regr, dataSet)
    printCoeffsForRegression(regr, featureNamesList, featureNamesDict)


def main():

    alpha = 30.0

    labelToLetterDict = extractLabelToLetterDict(inSheet)
    featureDict = extractFeatureDict(labelToLetterDict, inSheet)
    averagesDict = extractAveragesDict(labelToLetterDict, inSheet)
    featureNamesList, featureNamesDict = extractFeatureNamesList(featureDict, averagesDict, inSheet)

    for featureName in featureNamesList:
        print featureName

    # Warning: modifies featureNamesDict
    dataSet = extractDataList(featureDict, averagesDict, featureNamesDict, inSheet)

    tp, tn, fp, fn = rotatingTraining(dataSet, 5, threshold=0.2, alpha=alpha)

    printStats(tp, tn, fp, fn)

    regr = linear_model.Ridge(alpha=alpha)

    train(regr, dataSet)
    printCoeffsForRegression(regr, featureNamesList, featureNamesDict)

def oldMain():

    labelToLetterDict = extractLabelToLetterDict(inSheet)
    featureDict = extractFeatureDict(labelToLetterDict, inSheet)
    averagesDict = extractAveragesDict(labelToLetterDict, inSheet)
    featureNamesList = extractFeatureNamesList(featureDict, averagesDict, inSheet)

    dataSet = extractDataList(featureDict, averagesDict, inSheet)

    random.shuffle(dataSet)

    numPoints = len(dataSet)
    trainingSet = dataSet[:numPoints/2] #/2
    validationSet = dataSet[numPoints/2:3*numPoints/4]
    testSet = dataSet[3*numPoints/4:]

    regr = linear_model.Ridge(alpha=1.0)
#    regr = SVC(kernel="linear")

    train(regr, trainingSet)

    printCoeffsForRegression(regr, featureNamesList)

#    train(regr, trainingSet)
    bestF1threshold, bestNoCancerThreshold = findOptimalThresholds(regr, validationSet)

    tpF1, tnF1, fpF1, fnF1 = test(regr, testSet, bestF1threshold)
    tpNC, tnNC, fpNC, fnNC = test(regr, testSet, bestNoCancerThreshold)

    print "Using optimal F1 threshold of", bestF1threshold, "over validation set:"
    print "F1 SCORE:", f1Score(tpF1, tnF1, fpF1, fnF1)
    print "True positives:", tpF1
    print "True negatives:", tnF1
    print "False positives:", fpF1
    print "False negatives:", fnF1
    print ""
    print "Using optimal no-cancer threshold of", bestNoCancerThreshold, "over validation set:"
    print "F1 SCORE:", f1Score(tpNC, tnNC, fpNC, fnNC)
    print "True positives:", tpNC
    print "True negatives:", tnNC
    print "False positives:", fpNC
    print "False negatives:", fnNC

    print len(dataSet)

wbIn = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME)
inSheet = wbIn.active

mainThreshold()
