import openpyxl
import string
import pickle

def convertToColumnLetter(x):
    lastDigit = string.ascii_uppercase[x % 26]

    dividend = x - (x%26)

    if dividend == 0:
        return lastDigit

    else:
        return convertToColumnLetter(dividend/26 - 1) + lastDigit

idToDataDict = {}

MAX_ROW = 1710
LAST_COLUMN_LETTER = "AH"

#SPREADSHEET_PATH = "/Users/adamyedidia/breast_cancer/spreadsheets/data_original.xlsx"
SPREADSHEET_PATH = "/Users/adamyedidia/breast_cancer/spreadsheets/image_mapper.xlsx"

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
sheet = wb.active

print "loaded sheet"

columnNameDict = {}

column = -1
columnLetter = "#" # not a real column letter

idColumn = None

while columnLetter != LAST_COLUMN_LETTER:
    column += 1
    columnLetter = convertToColumnLetter(column)

    columnName = sheet[columnLetter + "1"].value
    columnNameDict[columnLetter] = columnName

#    if columnName == "ID":
#        idColumn = columnLetter

    if columnName == "ACC REMATCH":
        idColumn = columnLetter

for rowNum in range(2, MAX_ROW+1):
    column = -1
    columnLetter = "#" # not a real column letter

    littleDict = {}

    while columnLetter != LAST_COLUMN_LETTER:
        column += 1
        columnLetter = convertToColumnLetter(column)

        if columnLetter == idColumn:

#            ID = sheet[columnLetter + str(rowNum)].value
            entry = sheet[columnLetter + str(rowNum)].value

            ID = entry[:string.find(entry, "-")]

        else:
            littleDict[columnNameDict[columnLetter]] = \
                sheet[columnLetter + str(rowNum)].value

    idToDataDict[str(ID)] = littleDict

    if rowNum % 1000 == 0:
        print "Read row", rowNum

pickle.dump(idToDataDict, open("spreadsheet_dict_new.p", "w"))
