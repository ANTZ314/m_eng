"""
Description:
	Kaggle dataset is too large to process with pre-set ViT dimensions
    Therefore must resize each image [224, 224, 3] to required [32, 32, 3]

STEPS:
    - Make new working directory
    - Copy 'train' & 'validation' images to same directory  - [600]
    - Split all images into patches [32, 32, 3]             - [29,400]
    - If too many images [44,039], move [14,639] into temp folder
    - 
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Break larger images into smaller tiles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
## New directories:
mkdir /path/patch/dataset/pinhole

## Copy both to single folder:
cp /path/kaggle/train/pinhole/*     /path/patch/dataset/pinhole/
cp /path/kaggle/validate/pinhole/*  /path/patch/dataset/pinhole/

"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## SPLIT IMAGES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from PIL import Image
from itertools import product
import os

# New dataset location (copy)
local = '/home/antz/Desktop/models/0_NOTES/patch/dataset/pinhole/'

# Check how many files in this directory (600 in each):
_, _, files = next(os.walk(local))
file_count = len(files)
print("File Count: {}".format(file_count))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Split original image + Remane & store new patch images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')

        img.crop(box).save(out)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# For each file in directory "dirX" "test" to "tiles":
d = 32
dir_out = local                         # images are stored in same directory
dir_in  = local                          # Repeat for each directory (dir1-dir6)

# List files in that directory
for filename in os.listdir(dir_in):
    f = os.path.join(dir_in, filename)
    # checking if it is a file
    if os.path.isfile(f):
        #print(f)                       # view each file created
        tile(f, dir_in, dir_out, d)     # split into blocks
        os.remove(f)                    # delete original

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Check Patches created (29,400)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
_, _, files = next(os.walk(dir_in))
file_count = len(files)
print("File Count: {}".format(file_count))





#========================================================================================#
## [OVERSIZED PATCH FOLDER] 
## Split oversized directory into smaller batches
## [44,039] = [29,400] + [14,639]
#========================================================================================#
# Move 14,639 files into separate 'temp' folder
import os
import shutil

# Make temp directory first

source = r'/home/antz/Desktop/models/0_NOTES/patch/dataset/pinhole/'          # files location
destination = r'/home/antz/Desktop/models/0_NOTES/patch/dataset/pinhole_temp' # where to move to

files = os.listdir(source)                          # returns a list with all the files in source

# 'n' files
for f in range(14639): #files:
    shutil.move(source + files[f], destination)
    #print(source + files[f])

print("complete")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Check split succesfully
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Check Patches created (29,400)
_, _, files = next(os.walk(source))
file_count = len(files)
print("Source Count: {}".format(file_count))

# Check Patches created (14,639)
_, _, files = next(os.walk(destination))
file_count = len(files)
print("Temp1 Count: {}".format(file_count))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# no need to copy to Drive
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

