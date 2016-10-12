import openpyxl
import string
# testing comment`
# Replaces 0's with blanks

dirtyColumns = ["R", "S", "T"]
cleanColumns = ["A", "B", "C"]

SHEET_NAME = "original_aug_1.xlsx"

inSheet = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME).active
wbOut = openpyxl.Workbook()
outSheet = wbOut.active

MAX_ROW = 1779

for a in range(1, MAX_ROW+1):
    for b, dirtyColumn in enumerate(dirtyColumns):
        cleanColumn = cleanColumns[b]
        
        dirtyValue = inSheet[dirtyColumn + str(a)].value
        
        if dirtyValue != 0 and dirtyValue != "0":
            outSheet[cleanColumn + str(a)].value = dirtyValue
            
wbOut.save("../spreadsheets/clean_columns.xlsx")      
