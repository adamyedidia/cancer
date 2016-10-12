import openpyxl
import string
import sys

letter = sys.argv[1]

PATH_COLUMNS = ["AJ", "AK", "AL", "AM", "AN"]

SHEET_NAME = "original_aug_1.xlsx"

inSheet = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME).active
wbOut = openpyxl.Workbook()
outSheet = wbOut.active

for pathColumn in PATH_COLUMNS:
    assert inSheet[pathColumn + "1"].value[0] == letter
    assert inSheet[pathColumn + "1"].value[1:6] == "XPATH"
    assert len(inSheet[pathColumn + "1"].value) == 6 or int(inSheet[pathColumn + "1"].value[6])
    
outSheet["A1"].value = letter + "XPATH_CONCAT"

MAX_ROW = 1779

for a in range(2, MAX_ROW+1):
    concats = ""
    
    pathColumn = PATH_COLUMNS[0]
    path = string.strip(inSheet[pathColumn + str(a)].value)
    
    if path != None:
        concats += path
    
    for pathColumn in PATH_COLUMNS[1:]:
        path = inSheet[pathColumn + str(a)].value
        
        if path != None:
            concats +=  "," + string.strip(path)
    
    outSheet["A"+str(a)].value = concats
    
wbOut.save("../spreadsheets/" + letter + "XPATH_concat.xlsx")      