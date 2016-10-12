import openpyxl
import string

fbcColumn = "I"

SHEET_NAME = "nice_sheet_aug_1.xlsx"

inSheet = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME).active
wbOut = openpyxl.Workbook()
outSheet = wbOut.active

assert inSheet[fbcColumn + "1"].value == "FBC"
outSheet["A1"].value = "FBC_1"
outSheet["B1"].value = "FBC_M"
outSheet["C1"].value = "FBC_NO"

MAX_ROW = 1779

FIRST_DEGREES = ["M", "S", "D"]
MEN = ["F", "B", "MN", "MAN", "HB", "NW", "PF", "MF", "SO", "CM", "PU"]
NOS = ["NO"]

for a in range(2, MAX_ROW+1):
    num1 = 0
    numM = 0
    numNo = 0
    
    contents = inSheet[fbcColumn + str(a)].value
    
    if contents != None:
        contentsSplit = [string.strip(i) for i in string.split(contents, ",")]
        
        print contentsSplit, [i in FIRST_DEGREES for i in contentsSplit]
        
        for code in contentsSplit:
                                    
            if code in FIRST_DEGREES:
                num1 += 1
            if code in MEN:
                numM += 1
            if code in NOS:
                numNo += 1
                
    outSheet["A" + str(a)].value = num1
    outSheet["B" + str(a)].value = numM
    outSheet["C" + str(a)].value = numNo
    
wbOut.save("../spreadsheets/family_history_cols.xlsx")            