import openpyxl
import string

SXRESULT_COLUMN = "AJ"
NEXTMAM_COLUMNS = ["AR", "AU"]

SHEET_NAME = "nice_sheet_aug_1.xlsx"

inSheet = openpyxl.load_workbook("../spreadsheets/" + SHEET_NAME).active
wbOut = openpyxl.Workbook()
outSheet = wbOut.active

assert inSheet[SXRESULT_COLUMN + "1"].value == "SXRESULT"

for nextmamColumn in NEXTMAM_COLUMNS:
    assert inSheet[nextmamColumn + "1"].value[-1] == "A"
    assert inSheet[nextmamColumn + "1"].value[:7] == "NEXTMAM"

outSheet["A1"].value = "CANCER"

MAX_ROW = 1779

for a in range(2, MAX_ROW+1):
    sxresult = inSheet[SXRESULT_COLUMN + str(a)].value 
    
    if sxresult == None:
        for nextmamColumn in NEXTMAM_COLUMNS:
            nextmam = inSheet[nextmamColumn + str(a)].value
            if nextmam == 1 or nextmam == 2 or nextmam == "1" or nextmam == "2":
                outSheet["A" + str(a)].value = 0
                
    elif string.strip(sxresult) == "M":
        outSheet["A" + str(a)].value = 1
    elif string.strip(sxresult) == "H" or string.strip(sxresult) == "B":
        outSheet["A" + str(a)].value = 0
    
wbOut.save("../spreadsheets/outcomes.xlsx")      