import pickle
import os
import random
import sys
import string

print "Loading..."

monthDict = {"Jan": 1, \
            "Feb": 2, \
            "Mar": 3, \
            "Apr": 4, \
            "May": 5, \
            "Jun": 6, \
            "Jul": 7, \
            "Aug": 8, \
            "Sep": 9, \
            "Oct": 10, \
            "Nov": 11, \
            "Dec": 12}

# Returns True if the first one is first, and False if the first one is last
def compareTwoDateStrings(dateString1, dateString2):
    split1 = string.split(dateString1)
    split2 = string.split(dateString2)

    if int(split1[2]) < int(split2[2]):
        return True

    if int(split1[2]) > int(split2[2]):
        return False

    if monthDict[split1[0]] < monthDict[split2[0]]:
        return True

    if monthDict[split1[0]] > monthDict[split2[0]]:
        return False

    if int(split1[1][:-1]) < int(split2[1][:-1]):
        return True

    if int(split1[1][:-1]) > int(split2[1][:-1]):
        return False

    raise

def terminalReady(s):
    BAD_CHARS = [" ", "[", "]"]

    returnString = r""

    for char in s:
        if char in BAD_CHARS:
#            returnString += "\\"
            pass
        returnString += char

    returnString = string.replace(returnString, "a", "a")

    return returnString

idDict = pickle.load(open("id_dict.p", "r"))

IMAGE_PNG_PATH = "/Users/adamyedidia/breast_cancer/image_png/"

topLevelDirWeights = {"pngs_100k_to_110k": 3.2, \
                        "pngs_10k_to_20k": 2.3, \
                        "pngs_110k_plus": 1.1, \
                        "pngs_20k_to_30k": 2.7, \
                        "pngs_30k_to_40k": 2.5, \
                        "pngs_40k_to_100k": 16}

totalDirWeight = sum(topLevelDirWeights.values())

spreadsheetDict = pickle.load(open("spreadsheet_dict_new.p", "r"))

print 'Welcome. You will view mammogram images. Your goal is to guess attributes \
relating to the patient. Which attributes would you like to try to guess? Type \
"done" if you are done naming attributes.'

print "Possible attributes:"
possibleAttributes = spreadsheetDict["2175"].keys()

print possibleAttributes

listOfAttributes = []

while True:
    attribute = raw_input("-->")
    if attribute == "done":
        break
    else:
        if attribute in possibleAttributes:
            listOfAttributes.append(attribute)
        else:
            print "Sorry, that attribute doesn't exist."


while True:
    weightSoFar = 0.0
    randomDirValue = random.random() * totalDirWeight

    for dirName in topLevelDirWeights:
        weightSoFar += topLevelDirWeights[dirName]
        if weightSoFar > randomDirValue:
            break

    pathSoFar = IMAGE_PNG_PATH + dirName + "/"

    pseudoID = random.choice(os.listdir(pathSoFar))

    if pseudoID in idDict:
        ID = idDict[pseudoID]

        if ID in spreadsheetDict:
            pathSoFar = pathSoFar + pseudoID + "/"

            dates = os.listdir(pathSoFar)

#            print dates

            latestDate = dates[0]

            for date in dates[1:]:
                if compareTwoDateStrings(latestDate, date):
                    latestDate = date

#            print terminalReady(terminalReady(latestDate))

            os.chdir(pathSoFar)

            xrays = os.listdir(latestDate)

            print xrays

            os.chdir(latestDate)

            for xray in xrays:
                if ("R CC" in xray) or ("L CC" in xray) or ("R MLO" in xray) or \
                    ("L MLO" in xray):

                    imageName = os.listdir(xray)[0]
                    os.chdir(xray)
                    os.system("open " + imageName)
                    os.chdir("..")

            os.chdir(IMAGE_PNG_PATH)

            for attribute in listOfAttributes:
                print "Guess attributes of patient", ID
                print attribute
                raw_input("-->")
                print "Correct answer was", spreadsheetDict[ID][attribute]

        else:
            print "ID", ID, "missing from spreadsheet. Please investigate!"

    else:
        print "DicomName", pseudoID, "missing from dictionary. Please investigate!"
