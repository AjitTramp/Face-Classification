import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
dirName= "/media/ajit/5E64-0F5C/Humpback"
from subprocess import check_output
print(check_output(["ls",dirName]).decode("utf8"))

os.chdir(dirName)

# Get the list of training files 
train = os.listdir(dirName+'/train')
# Get the list of test files
test = os.listdir(dirName+'/test')

print("Total number of training images: ",len(train))
print("Toal number of test images: ",len(test))



sample = pd.read_csv(dirName+'/sample_submission.csv')
print(sample.shape)
sample.head()