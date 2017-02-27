import pickle
import openpyxl
import string

wb = openpyxl.load_workbook("../spreadsheets/image_mapper.xlsx")

sheet = wb.active

ID_COL = "A"
REMATCH_COL = "F"

IDToRematch = {}
rematchToID = {}

MAX_ROW = 1710

for rowNum in range(1, MAX_ROW+1):
    ID = str(sheet[ID_COL + str(rowNum)].value)
    rawRematch = str(sheet[REMATCH_COL + str(rowNum)].value)
    rematch = rawRematch[:string.find(rawRematch, "-")]

    IDToRematch[ID] = rematch
    rematchToID[rematch] = ID

pickle.dump(IDToRematch, open("id_to_rematch.p", "w"))
pickle.dump(rematchToID, open("rematch_to_id.p", "w"))
