import openpyxl
import string
import pickle

SPREADSHEET_PATH = "/Users/adamyedidia/breast_cancer/spreadsheets/image_mapper.xlsx"

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
sheet = wb.active

MAX_ROW = 4

IMGREPORT_COL = "AW"
ACC_REMATCH_COL = "F"

idToWhichBreastDict = {}

rowNum = 2

MIN_DIFF_TOL = 3

while True:
    try:

        entry = sheet[ACC_REMATCH_COL + str(rowNum)].value

        ID = entry[:string.find(entry, "-")]

        report = sheet[IMGREPORT_COL + str(rowNum)].value

        if "LEFT BREAST MAMMOGRAM" in report:
            idToWhichBreastDict[ID] = "L"
            rowNum += 1
        elif "RIGHT BREAST MAMMOGRAM" in report:
            idToWhichBreastDict[ID] = "R"
            rowNum += 1
        else:
            rightCount = report.count("right breast")
            leftCount = report.count("left breast")

            if rightCount > leftCount + MIN_DIFF_TOL:
                idToWhichBreastDict[ID] = "R"
                rowNum += 1
            elif leftCount > rightCount + MIN_DIFF_TOL:
                idToWhichBreastDict[ID] = "L"
                rowNum += 1
            else:
                print report
                result = raw_input(str(ID) + " -->")
                if result == 'l' or result == 'L':
                    idToWhichBreastDict[ID] = "L"
                    rowNum += 1
                elif result == 'r' or result == 'R':
                    idToWhichBreastDict[ID] = "R"
                    rowNum += 1
                elif result == 'b' or result == 'B':
                    idToWhichBreastDict[ID] = "B"
                    rowNum += 1
                elif result == 'done':
                    break
                else:
                    print "no"

                if rowNum == MAX_ROW:
                    break

    except:
        break


pickle.dump(idToWhichBreastDict, open("which_breast.p", "w"))
