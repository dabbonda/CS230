"""
Decompresses .dcm files within the Fetal directory.
"""

import gdcm
import sys
import os
from pprint import pprint



# reads from /data/FetalLung and writes to /home/mazin/FetalLung
def decompress_file(original_file):
    file_path, ext = original_file.split('.')
    decompressed_file = '-d.'.join([file_path, ext])
    decompressed_file = decompressed_file.replace('data', 'home/mazin')

    r = gdcm.ImageReader()
    r.SetFileName(original_file)
    if not r.Read():
        print("error occurred, one file was unreadable: {}".format(original_file))
        unprocessed.append(tuple((original_file, 'READ')))
        return


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

#     print(decompressed_file)
#     w.SetFileName('/home/mazin/temp.dcm')
    w.SetFileName(decompressed_file)
    w.SetFile(r.GetFile())
    w.SetImage(image)
    if not w.Write():
        print("error occurred, one file was unwritable: {}".format(original_file))
        unprocessed.append(tuple((original_file, 'WRITE')))
        return

unprocessed = []
# for root, subdirs, files in os.walk(os.path.join(os.getenv('HOME'), 'FetalLung')):
for root, subdirs, files in os.walk('/data/FetalLung'):
    print(root)
    if not files:
        continue
    for file in files:
        if not file.endswith('.dcm'):
            continue
        
        file_path = os.path.join(root,file)
        decompress_file(file_path)

        
# decompress_file('/data/FetalLung/FetalLungNormal/LC20150615260/COR SSFSExr BODY/IM-0080-0006.dcm')
pprint(unprocessed)
