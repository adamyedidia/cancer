import string
import dicom
import openpyxl
import os

#spreadsheetDict = pickle.load(open("spreadsheet_dict_new.p", "r"))

#print spreadsheetDict

DATA_SET_PATH = "/Users/adamyedidia/breast_cancer/EMI-ML/Mammography/Collaboration_Data_Set/"

SPREADSHEET_PATH = "/Users/adamyedidia/breast_cancer/spreadsheets/image_mapper.xlsx"

print "Loading..."

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
sheet = wb.active

print "Loaded sheet"

ACC_REMATCH_COL = "F"
AGE_COL = "C"

MAX_ROW = 1710
numDiscrepancies = 0

DIFF_TOL = 0

listOfARDirs = os.listdir(DATA_SET_PATH)

for rowNum in range(2, MAX_ROW+1):

    accRematch = sheet[ACC_REMATCH_COL + str(rowNum)].value
    spreadsheetAge = sheet[AGE_COL + str(rowNum)].value
    arFirstNumber = accRematch[:string.find(accRematch, "-")]

    arDir = arFirstNumber + "_" + accRematch
    if arDir in listOfARDirs:
        dicomPath = DATA_SET_PATH + arDir + "/"

        dicomOfChoice = os.listdir(dicomPath)[0]

        ds = dicom.read_file(dicomPath + dicomOfChoice)
        dicomAge = int(ds.PatientAge[:-1])

        if abs(dicomAge - spreadsheetAge) > DIFF_TOL:
            print "DISCREPANCY: Spreadsheet has age " + str(spreadsheetAge) + \
                " but DICOM has age " + str(dicomAge) + "."
            numDiscrepancies += 1

    else:
        print "Missing patient:", arDir

print "Number of discrepancies:", numDiscrepancies

#    age = spreadsheetDict[ID]["AGE"]



#for key in spreadsheetDict:
#    print key, spreadsheetDict[key]
