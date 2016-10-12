import openpyxl
import string

POC_COLUMN = "M"

SHEET_NAME = "nice_sheet_aug_1.xlsx"

inSheet = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME).active
wbOut = openpyxl.Workbook()
outSheet = wbOut.active

assert inSheet[POC_COLUMN + "1"].value == "POC"
outSheet["A1"].value = "POC_COUNT"

MAX_ROW = 1779

for a in range(2, MAX_ROW+1):
    count = 0
    
    contents = inSheet[POC_COLUMN + str(a)].value
    
    if contents != None:
        contentsSplit = [string.strip(i) for i in string.split(contents, ",")]
    
        count = len(contentsSplit)
        
    outSheet["A"+str(a)].value = count
    
wbOut.save("../spreadsheets/previous_cancer_cols.xlsx")      