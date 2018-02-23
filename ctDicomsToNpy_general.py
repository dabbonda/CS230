#!/usr/bin/env python
"""
Like ctDicomsToNpy.py except more general.
"""

import glob, os, sys, shutil, dicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lxml import etree

def main():
    projectPath = './data'
    subjectFolderPaths = getSubjectPathsForProject(projectPath)
    for subjectPath in subjectFolderPaths:
        convertCTDicomsToNpyForSubject(subjectPath)

def convertCTDicomsToNpyForSubject(subjectPath):
    ctSeriesPath, = getSeriesPathsForSubject(subjectPath)

    outputFilePath = seriesPathToNumpyFilePath(ctSeriesPath, "allCTSlices")

    # If the file exists, ask the user whether they want to overwrite.
    # Exit out of the function without saving if they do not want to overwrite.
    if (len(sys.argv) <= 1 or sys.argv[1] != '--overwrite') and os.path.exists(outputFilePath):
        queryQuestion = "File %s already exists. Would you like to overwrite it?" % outputFilePath
        try:
            if not query_yes_no(queryQuestion):
                return
        except EOFError as e:
            pass

    ctNumpyMatrix = ctSeriesPathToNumpyMatrix(ctSeriesPath)

    np.save(outputFilePath, ctNumpyMatrix)
    print("Saved %s" % outputFilePath) 
    print()

def getSubjectPathsForProject(projectPath):
    return glob.glob("%s/Subject-*" % projectPath)

def get_dcm_dict(dcm_path):
    """
    raw is the raw output from the dicom file, lung_im is the grayscale array
    ds is the original ds format if you want to work directly with it
    """
    ds = dicom.read_file(dcm_path)
    lung = ds.pixel_array
    lung_raw = np.asarray(lung, dtype='float')
    lung_im = (lung_raw/lung_raw.max()) * 255.0
    return {'raw': lung_raw, 'grayscale': lung_im, 'ds': ds}

def load_dicoms(ctDicomPaths, displacement):
    num_dicoms = len(ctDicomPaths)
    dcm_dict = get_dcm_dict(ctDicomPaths[0])
    ds = dcm_dict['ds']

    # Create the structure for the 3d volume
    data = np.zeros([num_dicoms, ds.Rows, ds.Columns])

    for path in ctDicomPaths:
        dcm_dict = get_dcm_dict(path)
        ds = dcm_dict['ds']
        slice_num = int(ds.InstanceNumber) - displacement
        if slice_num >= len(data):
            print("slice_num: ", slice_num )
            slice_num = len(data) - 1
        data[slice_num, :, :] = dcm_dict['raw']
        
    data = np.asarray(data, dtype='int16')
    
    # tumorSliceBitmap = data[10]
    # plt.imshow(tumorSliceBitmap, cmap=cm.Greys_r)
    # plt.show()

    return data

def ctSeriesPathToNumpyMatrix(ctSeriesPath):
    ctDicomPaths = getDicomPathsForCTSeriesPath(ctSeriesPath)

    displacement, maxFrameNum = ctSeriesPathToMinMaxFrameNumber(ctSeriesPath)

    print(displacement)
    print(maxFrameNum)
    print()

    # raw_matrix is the 3d array of the dicom slices in order
    raw_matrix = load_dicoms(ctDicomPaths, displacement)
    return raw_matrix

def getDicomPathsForCTSeriesPath(ctSeriesPath):
    dicomPaths = glob.glob("%s/*.dcm" % ctSeriesPath)
    return dicomPaths

def seriesPathToNumpyFilePath(seriesPath, fileNamePrefix):
    subjectName = pathToSubjectName(seriesPath)
    outputFileName = "%s_%s.npy" % (fileNamePrefix, subjectName)
    outputFilePath = os.path.join(seriesPath, outputFileName)
    return outputFilePath

def pathToSubjectName(path):
    return path.split('Subject-')[-1].split('/')[0]

def getSeriesPathsForSubject(subjectPath):
    seriesPaths = glob.glob("%s/COR-SSFSE-BODY -*" % subjectPath)
    return seriesPaths
    
def getCorrespondingCTFrameNumbersForSegDicom(segDicomPath):
    ctSeriesPath = segDicomPathToCTSeriesPath(segDicomPath)
    ctDicomPaths = getDicomPathsForCTSeriesPath(ctSeriesPath)
    ID2PathDict = {dicomPathToID(ctDicomPath): ctDicomPath for ctDicomPath in ctDicomPaths}

    correspondingCTImageIDs = segDicomToListOfCorrespondingCTImageIDs(segDicomPath)

    ctFrameNumbers = []
    for imageID in correspondingCTImageIDs:
        ctDicomPath = ID2PathDict[imageID]
        ctFrameNumber = getCTFrameNumberForCTDicomPath(ctDicomPath)
        ctFrameNumbers.append(int(ctFrameNumber))

    # Note: Might not want this sorted...
    return sorted(ctFrameNumbers)

def dicomPathToID(dicomPath):
    return dicomPath.split('/')[-1].replace('.dcm', '')

def ctSeriesPathToMinMaxFrameNumber(ctSeriesPath):
    ctDicomPaths = getDicomPathsForCTSeriesPath(ctSeriesPath)

    ctFrameNumbers = []
    for ctDicomPath in ctDicomPaths:
        ctFrameNumber = getCTFrameNumberForCTDicomPath(ctDicomPath)
        ctFrameNumbers.append(int(ctFrameNumber))
    return min(ctFrameNumbers), max(ctFrameNumbers)

def segDicomToListOfCorrespondingCTImageIDs(segDicomPath):
    dso = dicom.read_file(segDicomPath)
    sfgs = dso.SharedFunctionalGroupsSequence[0]
    dis = sfgs.DerivationImageSequence[0]
    sis = dis.SourceImageSequence
    CTImageIDs = [sourceImage.RefdSOPInstanceUID for sourceImage in sis]
    return CTImageIDs

# subjectPath = pathToSubjectPath(segDicomPath)
def pathToSubjectPath(path):
    while True:
        folderName = path.split('/')[-1]
        if folderName.startswith('Subject-'):
            break
        path = os.path.dirname(path)
    return path

def segDicomPathToCTSeriesPath(segDicomPath):
    subjectPath = pathToSubjectPath(segDicomPath)
    seriesPaths = getSeriesPathsForSubject(subjectPath)
    coreSegSeriesPath, ctSeriesPath = unsortedSeriesPathPairToSegThenCTPathPair(seriesPaths)
    return ctSeriesPath

def getCTFrameNumberForCTDicomPath(ctDicomPath):
    dicom_obj = dicom.read_file(ctDicomPath)
    if False:
        print("FrameOfReferenceUID: ", dicom_obj.FrameOfReferenceUID) 
        print("InstanceNumber: ", dicom_obj.InstanceNumber) 
        print("SliceLocation: ", dicom_obj.SliceLocation) 
        print("WindowCenter: ", dicom_obj.WindowCenter) 
        print("WindowWidth: ", dicom_obj.WindowWidth) 
    ctFrameNumber = dicom_obj.InstanceNumber
    return ctFrameNumber 

def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def listPrint(li):
    for elem in li:
        print(elem) 

def listEqualQ(li1, li2):
    if len(li1) != len(li2):
        return False
    for elem1, elem2 in zip(li1, li2):
        if elem1 != elem2:
            return False
    return True

if __name__ == '__main__':
    main()