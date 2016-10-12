import pickle
import dicom
import os

DICOM_PATH = "/Users/adamyedidia/breast_cancer/EMI-ML/Mammography/Collaboration_Data_Set/"

dicomDirs = os.listdir(DICOM_PATH)

dictionary = {}

for dicomDir in dicomDirs:
    if dicomDir != ".DS_Store":

        innerDCMs = os.listdir(DICOM_PATH + dicomDir)

#        print "scraping", dicomDir

        for innerDCM in innerDCMs:
            if innerDCM[-4:] == ".dcm":
                ds = dicom.read_file(DICOM_PATH + dicomDir + "/" + innerDCM)
                break

        if ds.PatientsName in dictionary and dictionary[ds.PatientsName] != \
            ds.PatientID:
            print "ALERT: duplicate entry for pseudoID", ds.PatientsName
            print "Erasing ID", dictionary[ds.PatientsName]
            print "Replacing with", ds.PatientID
        dictionary[ds.PatientsName] = ds.PatientID

pickle.dump(dictionary, open("id_dict.p", "w"))
