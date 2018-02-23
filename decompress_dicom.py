"""
Decompresses .dcm files within the Fetal directory.
"""

import gdcm
import sys
import os

# change to FetalLungNormal and get rid of the /patient part in both areas if trying to decompress the normal ones
patients = os.listdir("/data/FetalLung/FetalLungAbnormal/patient")
# patients = os.listdir("/data/FetalLung/FetalLungNormal")
for folder in patients:
    read_dir = "/data/FetalLung/FetalLungAbnormal/patient/" + folder + "/COR SSFSE BODY/"
    # read_dir = "/data/FetalLung/FetalLung/" + folder + "/COR SSFSE BODY/"
    try:
        os.chdir(read_dir)
    except: # for some reason some of the patients have ssfsex instead?
        continue
    dicomms = os.listdir(os.getcwd())
    for filename in dicomms:
        if filename.endswith(".dcm"):
            # print(os.path.join(directory, filename))
            os.chdir(read_dir)# have to change back since we changed to write
            file1 = filename
            file2 = filename.split(".")[0] + "-d.dcm"

            r = gdcm.ImageReader()
            r.SetFileName(file1)
            if not r.Read():
                print "error occurred, one file was unreadable"
                sys.exit(1)

            os.chdir(os.getenv("HOME"))

            ir = r.GetImage()
            w = gdcm.ImageWriter()
            image = w.GetImage()

            image.SetNumberOfDimensions(ir.GetNumberOfDimensions())
            dims = ir.GetDimensions()


            image.SetDimension(0, ir.GetDimension(0))
            image.SetDimension(1, ir.GetDimension(1))

            pixeltype = ir.GetPixelFormat()
            image.SetPixelFormat(pixeltype)

            pi = ir.GetPhotometricInterpretation()
            image.SetPhotometricInterpretation(pi)

            pixeldata = gdcm.DataElement(gdcm.Tag(0x7fe0, 0x0010))
            str1 = ir.GetBuffer()
            # print ir.GetBufferLength()
            pixeldata.SetByteValue(str1, gdcm.VL(len(str1)))
            image.SetDataElement(pixeldata)

            w.SetFileName(file2)
            w.SetFile(r.GetFile())
            w.SetImage(image)
            if not w.Write():
                sys.exit(1)
        else:
            pass
os.chdir(os.getenv("HOME"))
os.system("rm -r -f decompressed; mkdir decompressed; mv *-d.dcm decompressed")
