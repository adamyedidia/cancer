from sklearn import ensemble
from nice_linear_regressor import *
import numpy as np

MAX_ROW = 1779

#LIST_OF_BINARY_COLS = ["RACE","PBC","SMOKE","ALCOHOL",
#    "ASHKENAZI","HORMONE","ASSESS","DENSITY","FIRSTMAM","FROMSCREEN", "HORMONE",
#    "FINDING","FINDTYPE","BXTYPE","BXPATH_CONCAT"]

LIST_OF_BINARY_COLS = []

#LIST_OF_REAL_COLS = ["AGE","FBC_1","FBC_M","FBC_NO","POC_COUNT","WEIGHT",
#    "HEIGHT","MALE","BIRTH","AGEPREG","MAAGE","MPAGE","PRIORBX"]    

LIST_OF_REAL_COLS = []

REPORT_COL = "BXPATHREP"

LIST_OF_EXCLUDED_COLS = ["ID","PACC","SDATE","STUDY","LOC","SPROCS","BXRESULT",
    "BXDATE"]
    
SHEET_NAME = "nice_sheet_aug_1.xlsx"

def addToNumInstances(numInstancesDict, v):
    if v in numInstancesDict:
        numInstancesDict[v] += 1
    else:
        numInstancesDict[v] = 1

def rotatingTraining(dataSet, rotationSize, threshold=0.2):    
    
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
        
        print "Test set starting at point", i
    
        tp, tn, fp, fn = trainAndTest(trainingSet, testSet, threshold)
        
        totalTP += tp
        totalTN += tn
        totalFP += fp
        totalFN += fn
        
        i += rotationSize
        
    return totalTP, totalTN, totalFP, totalFN    
    
def trainAndTest(trainingSet, testSet, threshold):
    regr = ensemble.ExtraTreesClassifier()
    
    train(regr, trainingSet)
        
    tp, tn, fp, fn = test(regr, testSet, threshold)
    
    return tp, tn, fp, fn    

def train(regr, trainingSet):
    x, y = turnDataListIntoTwoArrays(trainingSet)
    regr.fit(x, np.ravel(y))

def listOfBigramifyAndUnigramify(report):
    
    strippedReport = stripOfPunctuation(report)
    words = string.split(strippedReport)
        
    lastWord = words[0]
    
    listOfBigrams = []
    
    for word in words[1:]:  
        listOfBigrams.append((lastWord, word))
        
        lastWord = word
    
    listOfUnigrams = words    
        
    return listOfBigrams + listOfUnigrams

def extractFeatureNamesList(featureDict, averagesDict, commonBigrams, inSheet):
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
    
        elif columnLabel == REPORT_COL:
            for gram in commonBigrams:
                featureName = columnLabel + "_" + str(gram)
                
                featureNamesList.append(featureName)
                featureNamesDict[featureName] = [0, 0]
                
        else:
            raise        
    
        columnNum += 1
        columnLetter = convertToColumnLetter(columnNum)
        columnLabel = getColumnLabel(columnLetter, inSheet)
        
    return featureNamesList, featureNamesDict

def extractBigrams(sheet):
    bigramDict = {}
    
    for rowNum in range(2, MAX_ROW+1):
        report = sheet[REPORT_COL + str(rowNum)].value
        
        if report:
            listOfBigramsAndUnigrams = listOfBigramifyAndUnigramify(report)
    
            for gram in listOfBigramsAndUnigrams:
                addToNumInstances(bigramDict, bigram)
            
    return bigramDict
    
def pruneRareFeatures(dic, threshold):
    choppingBlock = []
    
    for key in dic:
        if dic[key] <= threshold:
            choppingBlock.append(key)
            
    for key in choppingBlock:
        del dic[key]    
    
    
bigramDict = extractBigrams(inSheet)
pruneRareFeatures(bigramDict, 5)
commonBigrams = bigramDict.keys()

dataSet = vectorify(commonBigrams, inSheet)

random.shuffle(dataSet)
    
labelToLetterDict = extractLabelToLetterDict(inSheet)
featureDict = extractFeatureDict(labelToLetterDict, inSheet)
averagesDict = extractAveragesDict(labelToLetterDict, inSheet)
featureNamesList, featureNamesDict = extractFeatureNamesList(featureDict, averagesDict, inSheet)  

dataSet = extractDataList(featureDict, averagesDict, featureNamesDict, inSheet)

tp, tn, fp, fn = rotatingTraining(dataSet, 5, threshold=0.2)

#printStats(tp, tn, fp, fn)

#regr = ensemble.ExtraTreesClassifier()

#train(regr, dataSet)
#printCoeffsForRegression(regr, featureNamesList, featureNamesDict)
