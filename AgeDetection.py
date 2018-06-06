data_dir='/home/ajit/Desktop/AgeDetection'

% pylab inline
import os
import random

import pandas as pd
from scipy.misc import imread

root_dir = os.path.abspath('.')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
