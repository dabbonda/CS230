import gdcm
import sys
import os


for filename in os.listdir(os.getcwd()):
    if filename.endswith(".dcm"):
        # print(os.path.join(directory, filename))

        file1 = filename
        file2 = filename.split(".")[0] + "-d.dcm"

        r = gdcm.ImageReader()
        r.SetFileName(file1)
        if not r.Read():
            print "error occurred, one file was unreadable"
            sys.exit(1)

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
os.system("rm -r -f decompressed; mkdir decompressed; mv *-d.dcm decompressed")