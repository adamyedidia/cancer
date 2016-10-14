import openpyxl
import string
import pickle

SPREADSHEET_PATH = "/Users/adamyedidia/breast_cancer/spreadsheets/image_mapper.xlsx"

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
sheet = wb.active

MAX_ROW = 1710

IMGREPORT_COL = "AW"
ACC_REMATCH_COL = "F"

idToWhichBreastDict = {}

rowNum = 2

numLBMs = 0
numRBMs = 0
numBLMs = 0

#MIN_DIFF_TOL = 3

while True:
    entry = sheet[ACC_REMATCH_COL + str(rowNum)].value

    ID = entry[:string.find(entry, "-")]

    report = sheet[IMGREPORT_COL + str(rowNum)].value

    if "LEFT BREAST MAMMOGRAM" in report:
        idToWhichBreastDict[ID] = "L"
        rowNum += 1
        numLBMs += 1

    elif "RIGHT BREAST MAMMOGRAM" in report:
        idToWhichBreastDict[ID] = "R"
        rowNum += 1
        numRBMs += 1

    elif "BILATERAL MAMMOGRAM" in report:
        rowNum += 1
        numBLMs += 1

    else:
        print "alert"
        rowNum += 1

    print rowNum, numLBMs, numRBMs, numBLMs

    if rowNum == MAX_ROW:
        break



print "Bilateral mammograms:", numBLMs
print "Left breast mammograms:", numLBMs
print "Right breast mammograms:", numRBMs

#pickle.dump(idToWhichBreastDict, open("which_breast.p", "w"))
