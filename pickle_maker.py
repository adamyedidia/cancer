import pickle
import random
import sys
import openpyxl
import string

try:
    SHEET_NAME = sys.argv[1]
except:
    raise Exception("\n\nUsage: python word_processing.py [spreadsheet name].xlsx\n" + \
        "Puts the processed spreadsheet in [spreadsheet name]_processed.xlsx")

wbIn = openpyxl.load_workbook(SHEET_NAME)
inSheet = wbIn.active
    
REPORT_COL = "P"
RESULT_COL = "Q"

MAX_ROW = 437

dataList = []

for row in range(2, MAX_ROW):
    dataList.append([])
    dataList[-1].append(inSheet[REPORT_COL + str(row)].value)
    dataList[-1].append(string.strip(inSheet[RESULT_COL + str(row)].value))
    
numPoints = len(dataList)    
trainingSet = dataList[:int(3.*numPoints/4.)]
testSet = dataList[int(3.*numPoints/4.):]  
    
pickle.dump(trainingSet, open("adams_training_set.p", "wb"))
pickle.dump(testSet, open("adams_test_set.p", "wb"))