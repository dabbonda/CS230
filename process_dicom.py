"""Split the FETAL dataset into train/val/test.

The FETAL dataset comes into the following format:
FETAL/
    decompressed-abnormal/
        IM-0263-0006-d.dcm
        ...
    decompressed-normal/
        IM-0272-0008-d.dcm
        ...

Original images have size (512, 512).

# Resizing to (64, 64) reduces the dataset size, and loading smaller images makes training faster.
# We already have a test set created, so we only need to split "train" into train and val sets.
# Because we don't have a lot of images and we want that the statistics on the val set be as
# representative as possible, we'll take 20% of "train" as val set.
"""

import glob, os, sys, shutil, dicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lxml import etree

DEBUG = True

# Directory holding the decompressed fetal images
input_directory = './data/FETAL'

# Directory holding the processed .dcm files
output_directory = './data/FETAL/processed'

def getPathsForClasses(input_directory):
    """
    Takes in the path to the directory containing the class folders.

    Returns the path for each class.
    """
    return glob.glob("%s/decompressed-*" % input_directory)

def getUniqueScanIDs(class_path):
    """
    Each .dcm file is in the format IM-[scan_id]-[series #].dcm 
    (e.g. IM-1126-0010.dcm)

    Returns a list of unique scan_ids.
    """
    unique_ids = set()
    dicom_paths = glob.glob("%s/*.dcm" % class_path)
    for dicom_path in dicom_paths:
        scan_id = dicom_path.split('-')[-3]
        unique_ids.add(scan_id)
    return list(unique_ids)

def convertDicomsToMatrix(class_label, class_path, scan_id):
    """
    Converts and saves the .dcm files to .npy numpy matrix files in the 
    class directory.
    """
    output_file_path = getOutputFilePath(class_label, scan_id)

    # If the file exists, ask the user whether they want to overwrite.
    # Exit out of the function without saving if they do not want to overwrite.
    if (len(sys.argv) <= 1 or sys.argv[1] != '--overwrite') and os.path.exists(output_file_path):
        try:
            query_question = "File %s already exists. Would you like to overwrite it?" % output_file_path
            if not query_yes_no(query_question):
                return
        except EOFError as e:
            pass

    getNumpyMatrix(class_path, scan_id)

    # Convert to 3D numpy matrix and save to output directory
    numpy_matrix = getNumpyMatrix(class_path, scan_id)
    np.save(output_file_path, numpy_matrix)

    print("Saved %s" % output_file_path)
    print()

def getNumpyMatrix(class_path, scan_id):
    """
    Collects .dcm files into a 3-D numpy matrix.
    """
    # Get the dicom paths for this scan_id
    dicom_paths = getDicomPathsForScanID(class_path, scan_id)

    displacement, max_frame_num = seriesPathToMinMaxFrameNumber(dicom_paths)
    if DEBUG:
        print(displacement)
        print(max_frame_num)
        print()

    # raw_matrix is the 3d array of the dicom slices in order
    raw_matrix = load_dicoms(dicom_paths, displacement)
    return raw_matrix

def getDicomPathsForScanID(class_path, scan_id):
    """
    Looks in the class directory for all .dcm files matching
    the scan ID.
    """
    dicom_paths = glob.glob("%s/IM-%s*.dcm" % (class_path, scan_id))
    return dicom_paths

def seriesPathToMinMaxFrameNumber(dicom_paths):
    allFrameNumbers = []
    for dicom_path in dicom_paths:
        frameNumber = getFrameNumberForDicomPath(dicom_path)
        allFrameNumbers.append(int(frameNumber))
    return min(allFrameNumbers), max(allFrameNumbers)

def getFrameNumberForDicomPath(dicom_path):
    dicom_obj = dicom.read_file(dicom_path)
    if DEBUG:
        print("FrameOfReferenceUID: ", dicom_obj.FrameOfReferenceUID) 
        print("InstanceNumber: ", dicom_obj.InstanceNumber) 
        print("SliceLocation: ", dicom_obj.SliceLocation) 
        print("WindowCenter: ", dicom_obj.WindowCenter) 
        print("WindowWidth: ", dicom_obj.WindowWidth) 
    frameNumber = dicom_obj.InstanceNumber
    return frameNumber

def load_dicoms(dicom_paths, displacement):
    num_dicoms = len(dicom_paths)
    dcm_dict = get_dcm_dict(dicom_paths[0])
    ds = dcm_dict['ds']

    # Create the structure for the 3-D volume
    data = np.zeros([num_dicoms, ds.Rows, ds.Columns])

    for dicom_path in dicom_paths:
        dcm_dict = get_dcm_dict(dicom_path)
        ds = dcm_dict['ds']
        slice_num = int(ds.InstanceNumber) - displacement
        if slice_num >= len(data):
            print("slice_num: ", slice_num )
            slice_num = len(data) - 1
        data[slice_num, :, :] = dcm_dict['raw']
        
    data = np.asarray(data, dtype='int16')
    
    # sliceBitmap = data[10]
    # plt.imshow(sliceBitmap, cmap=cm.Greys_r)
    # plt.show()

    return data

def get_dcm_dict(dicom_path):
    """
    lung_raw: the raw output from the dicom file
    lung_im: the grayscale array
    ds: the original ds format if you want to work directly with it
    """
    ds = dicom.read_file(dicom_path)
    lung = ds.pixel_array
    lung_raw = np.asarray(lung, dtype='float')
    lung_im = (lung_raw/lung_raw.max()) * 255.0
    return {'raw': lung_raw, 'grayscale': lung_im, 'ds': ds}

def getOutputFilePath(class_label, scan_id):
    """
    Returns the .npy file name with [label]-[scan_id].npy
    (e.g. 0-1127.npy)
    """
    outputFileName = "%s_%s.npy" % (class_label, scan_id)
    outputFilePath = os.path.join(output_directory, outputFileName)
    return outputFilePath

#### Utility Functions ####
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

if __name__ == '__main__':
    class_paths = getPathsForClasses(input_directory)
    for class_idx, class_path in enumerate(class_paths): # abnormal, normal
        scan_ids = getUniqueScanIDs(class_path) # get the unique scans in the class
        for scan_id in scan_ids:
            convertDicomsToMatrix(class_idx, class_path, scan_id)