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

#    print dateString1, dateString2
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

    return True

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
#NONE_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/dataset/train/None/"
#CALCS_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/dataset/train/CALCS/"
#NONE_DEV_PATH = "/Users/adamyedidia/breast_cancer/dataset/dev/None/"
#CALCS_DEV_PATH = "/Users/adamyedidia/breast_cancer/dataset/dev/CALCS/"

#ONE_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/train/1/"
#TWO_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/train/2/"
#THREE_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/train/3/"
#FOUR_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/train/4/"
#M_TRAIN_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/train/M/"

#ONE_TEST_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/dev/1/"
#TWO_TEST_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/dev/2/"
#THREE_TEST_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/dev/3/"
#FOUR_TEST_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/dev/4/"
#M_TEST_PATH = "/Users/adamyedidia/breast_cancer/density_dataset/dev/M/"

DATASET_PATH = "/Users/adamyedidia/breast_cancer/birads_dataset/"



def getPath(path, setType, category):
    return path + "/" + setType + "/" + category + "/"

topLevelDirWeights = { \
                        "pngs_100k_to_110k": 3.2, \
                        "pngs_10k_to_20k": 2.3, \
                        "pngs_110k_plus": 1.1, \
                        "pngs_20k_to_30k": 2.7, \
                        "pngs_30k_to_40k": 2.5, \
                        "pngs_40k_to_100k": 16}

totalDirWeight = sum(topLevelDirWeights.values())

spreadsheetDict = pickle.load(open("spreadsheet_dict_new.p", "r"))

imageCount = 0

attribute = "ASSESS"
TRAINING_PROB = 0.8

whichBreastDict = pickle.load(open("which_breast.p", "r"))

#print whichBreastDict

def putImageInDir(xray, imageCount, ID, attribute):
    imageName = os.listdir(xray)[0]
    uniqueImageName = str(ID) + "_" + str(imageCount) + "_" + imageName
    os.chdir(xray)

    imageCount += 1

    if random.random() < TRAINING_PROB:
        setType = "train"
    else:
        setType = "dev"


    if attribute == "unknown":
        attributeDir = "unknown"
    else:
        attributeDir = str(spreadsheetDict[ID][attribute])

    os.system("cp " + imageName + " " + getPath(DATASET_PATH, \
        setType, attributeDir) + uniqueImageName)

    os.chdir("..")

    return imageCount

for dirName in topLevelDirWeights:
    pathSoFar = IMAGE_PNG_PATH + dirName + "/"

    print "Combing directory", dirName

    for pseudoID in os.listdir(pathSoFar):
        if pseudoID in idDict:
            #print pseudoID
            ID = idDict[pseudoID]

            if ID in spreadsheetDict:
#                print "ID ok:", ID
                pathSoFar = IMAGE_PNG_PATH + dirName + "/" + pseudoID + "/"

                dates = os.listdir(pathSoFar)

                if ".DS_Store" in dates:
                    dates.remove(".DS_Store")

                latestDate = dates[0]

                for date in dates[1:]:
                    if compareTwoDateStrings(latestDate, date):
                        latestDate = date

                os.chdir(pathSoFar)
                xrays = os.listdir(latestDate)
                os.chdir(latestDate)

                if ID in whichBreastDict:
                    laterality = whichBreastDict[ID]
#                    print "Laterality ID", ID
                else:
                    laterality = None
                    print "No laterality ID:", ID


                for xray in xrays:
#                    if ("R CC" in xray) or ("L CC" in xray) or ("R MLO" in xray) or \
#                        ("L MLO" in xray):

                    if " MLO " in xray or " CC " in xray or " MLO-" in xray \
                        or " CC-" in xray:
#                    if True:

                        if laterality == "B":
                            if xray != ".DS_Store":
                                imageCount = putImageInDir(xray, imageCount, ID, attribute)

                        elif laterality == "R":
                            if " R " in xray:
                                imageCount = putImageInDir(xray, imageCount, ID, attribute)
                            elif xray != ".DS_Store":
                                imageCount = putImageInDir(xray, imageCount, ID, "unknown")

                        elif laterality == "L":
                            if " L " in xray:
                                imageCount = putImageInDir(xray, imageCount, ID, attribute)
                            elif xray != ".DS_Store":
                                imageCount = putImageInDir(xray, imageCount, ID, "unknown")

#                    else:
#                        print xray

                os.chdir(IMAGE_PNG_PATH)

            else:
                print "ID not in spreadsheet:", ID
                pass
        else:
            print "pseudoID not in dictionary:", pseudoID

print "TOTAL IMAGES", imageCount
