"""Processes the FETAL .dcm dataset into .npy files.

The FETAL dataset comes into the following format:
FetalLung/
	FetalLungNormal/ 
		LC20141215330/
			series/
				IM-0182-0001-d.dcm
				...
		LC20141217317/
			AX_SSFSE_BODY/
				IM-0991-0001-d.dcm
				...
			COR_SSFSE_BODY/
				IM-0991-0001-d.dcm
				...
			SAG_SSFSE_BODY/
				IM-0992-0001-d.dcm
				...
		...
	FetalLungAbnormal/ 
		patient/
			LC20150128274/
				AX_SSFSExr_BODY/
					IM-0782-0001-d.dcm
					...
				Cor_SSFSExr_BODY/
					IM-0781-0001-d.dcm
					...
				SAG_SSFSExr_BODY/
					IM-0783-0001-d.dcm
					...
			...



FETAL/
    decompressed-abnormal/
        IM-0263-0006-d.dcm
        ...
    decompressed-normal/
        IM-0272-0008-d.dcm
        ...

The output .npy files are placed in the following directory:
FETAL/
    processed/
        0-0002.npy
        ...
with each file as [class]-[scan_id].npy where class is 
0 (abnormal) or 1 (normal).
"""

import glob, os, sys, shutil, dicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lxml import etree

DEBUG = False

# Directory holding the decompressed normal fetal images
normal_directory = '/../home/mazin/FetalLung/FetalLungNormal'

# Directory holding the decompressed abnormal fetal images
abnormal_directory = '/../home/mazin/FetalLung/FetalLungAbnormal/patient'

# Directory output for the the processed .dcm files
output_directory = './data/FETAL/processed'

# def getPathsForClasses(input_directory):
#     """
#     Takes in the path to the directory containing the class folders.

#     Returns the path for each class.
#     """
#     return glob.glob("%s/FetalLung-*" % input_directory)

def getPathsForSubjects(class_path):
    subject_paths = glob.glob("%s/*" % class_path)
    subject_ids = [path.split('/')[-1] for path in subject_paths]
    return subject_ids, subject_paths

def getPathsForScans(subject_path):
    scan_paths = glob.glob("%s/*" % subject_path)
    scan_types = [path.split('/')[-1] for path in scan_paths]
    return scan_types, scan_paths

def getUniqueScanIDs(scan_path):
    """
    Each .dcm file is in the format IM-[scan_id]-[series #].dcm 
    (e.g. IM-1126-0010.dcm)

    Returns a list of unique scan_ids.
    """
    unique_ids = set()
    dicom_paths = glob.glob("%s/*.dcm" % scan_path)
    for dicom_path in dicom_paths:
        scan_id = dicom_path.split('-')[-3]
        unique_ids.add(scan_id)
    return list(unique_ids)

def convertDicomsToMatrix(class_label, scan_path, subject_id):
    """
    Converts and saves the .dcm files to .npy numpy matrix files in the 
    class directory.
    """
    output_file_path = getOutputFilePath(class_label, subject_id) # without orientation

    # Convert to 3D numpy matrix and save to output directory
    numpy_matrix = getNumpyMatrix(scan_path)

    # If the file exists, ask the user whether they want to overwrite.
    # Exit out of the function without saving if they do not want to overwrite.
    if (len(sys.argv) <= 1 or sys.argv[1] != '--overwrite') and os.path.exists(output_file_path):
        try:
            query_question = "File %s already exists. Would you like to overwrite it?" % output_file_path
            if not query_yes_no(query_question):
                return
        except EOFError as e:
            pass

    # np.save(output_file_path, numpy_matrix)

    print("Saved %s" % output_file_path)
    print()

def getNumpyMatrix(scan_path):
    """
    Collects .dcm files into a 3-D numpy matrix.
    """
    # Get the dicom paths for this scan_id
    dicom_paths = getDicomPathsForScanID(scan_path)

    displacement, max_frame_num = seriesPathToMinMaxFrameNumber(dicom_paths)
    if DEBUG:
        print(displacement)
        print(max_frame_num)
        print()

    # raw_matrix is the 3d array of the dicom slices in order
    print(scan_path)
    raw_matrix = load_dicoms(dicom_paths, displacement)
    return raw_matrix

def getDicomPathsForScanID(scan_path):
    """
    Looks in the scan directory for all .dcm files.
    """
    dicom_paths = glob.glob("%s/*.dcm" % scan_path)
    return dicom_paths

def seriesPathToMinMaxFrameNumber(dicom_paths):
    allFrameNumbers = []
    for dicom_path in dicom_paths:
        frameNumber = getFrameNumberForDicomPath(dicom_path)
        allFrameNumbers.append(frameNumber)
    return min(allFrameNumbers), max(allFrameNumbers)

def getFrameNumberForDicomPath(dicom_path):
    dicom_obj = dicom.read_file(dicom_path)
    if DEBUG:
        print("FrameOfReferenceUID: ", dicom_obj.FrameOfReferenceUID) 
        print("InstanceNumber: ", dicom_obj.InstanceNumber) 
        print("SliceLocation: ", dicom_obj.SliceLocation) 
        print("WindowCenter: ", dicom_obj.WindowCenter) 
        print("WindowWidth: ", dicom_obj.WindowWidth) #ImageOrientationPatient
    frameNumber = int(dicom_obj.InstanceNumber)
    return frameNumber

def load_dicoms(dicom_paths, displacement):
    num_dicoms = len(dicom_paths)
    dcm_dict = get_dcm_dict(dicom_paths[0])
    ref_ds = dcm_dict['ds']

    # Create the structure for the 3-D volume
    data = np.zeros([num_dicoms, ref_ds.Rows, ref_ds.Columns])

    for dicom_path in dicom_paths:
        dcm_dict = get_dcm_dict(dicom_path)
        ds = dcm_dict['ds']
        if ds.Rows != ref_ds.Rows or ds.Columns != ref_ds.Columns:
            print("Invalid scan size: " +  str([ds.Rows, ds.Columns]))
            continue    
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

def getOutputFilePath(class_label, subject_id):
    """
    Returns the .npy file name with [label]-[scan_id].npy
    (e.g. 0-1127.npy)
    """
    outputFileName = "%s_%s.npy" % (class_label, subject_id)
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
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

if __name__ == '__main__':
    class_paths = [normal_directory, abnormal_directory] # getPathsForClasses(...)
    for class_idx, class_path in enumerate(class_paths): # abnormal, normal
        subject_ids, subject_paths = getPathsForSubjects(class_path) 
        for subject_id, subject_path in zip(subject_ids, subject_paths): 
            scan_types, scan_paths = getPathsForScans(subject_path) # e.g. series/, COR/, AXL/, etc.
            for scan_type, scan_path in zip(scan_types, scan_paths):
                scan_ids = getUniqueScanIDs(scan_path) # get the unique scans in the class
                print(str(class_idx) + "-" + str(subject_id) + "-" + str(scan_type) + "-" + str(scan_ids))
                # convertDicomsToMatrix(class_idx, scan_path, subject_id)
