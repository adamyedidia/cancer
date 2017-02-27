import openpyxl

wb = openpyxl.load_workbook('../spreadsheets/manisha_new.xlsx')
mainSheet = wb.active

SXPATH_COLS = ["AS", "AT", "AU", "AV"]

wbOut = openpyxl.Workbook()
outSheet = wbOut.active

MAX_ROW = 1118
numInvasives = 0

for rowNum in range(2, MAX_ROW+1):

    invasive = "B"

    for sxpathCol in SXPATH_COLS:
        entry = mainSheet[sxpathCol + str(rowNum)].value
        if entry != None:
            if entry[0] == "I":
                invasive = "M"
                numInvasives += 1

    outSheet["A" + str(rowNum)].value = invasive

print numInvasives
wbOut.save("manisha_invasive_col.xlsx")
