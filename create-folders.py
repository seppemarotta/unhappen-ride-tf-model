import os
from utils import split_data

SOURCEDIR = '/Users/giuseppemarotta/Documents/raw-data/originals/'
print(len(os.listdir('/Users/giuseppemarotta/Documents/raw-data/originals/unhappen-rides')))
print(len(os.listdir('/Users/giuseppemarotta/Documents/raw-data/originals/valid-rides')))

try:
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/training')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/testing')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/training/unhappen')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/training/happen')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/testing/happen')
    os.mkdir(SOURCEDIR+'tmp/unhappen-v-happen/testing/unhappen')
except OSError:
    pass


UNHAPPEN_SOURCE_DIR = SOURCEDIR+"unhappen-rides/"
TRAINING_UNHAPPEN_DIR = SOURCEDIR+"tmp/unhappen-v-happen/training/unhappen/"
TESTING_UNHAPPEN_DIR = SOURCEDIR+"tmp/unhappen-v-happen/testing/unhappen/"
HAPPEN_SOURCE_DIR = SOURCEDIR+"valid-rides/"
TRAINING_HAPPEN_DIR = SOURCEDIR+"tmp/unhappen-v-happen/training/happen/"
TESTING_HAPPEN_DIR = SOURCEDIR+"tmp/unhappen-v-happen/testing/happen/"

split_size = .9

split_data(UNHAPPEN_SOURCE_DIR, TRAINING_UNHAPPEN_DIR, TESTING_UNHAPPEN_DIR, split_size)
split_data(HAPPEN_SOURCE_DIR, TRAINING_HAPPEN_DIR, TESTING_HAPPEN_DIR, split_size)

print("Training Unhappen: " + TRAINING_UNHAPPEN_DIR)
print("Training Happen: " + TRAINING_HAPPEN_DIR)

print("Testing Unhappen: " + TRAINING_UNHAPPEN_DIR)
print("Testing Happen: " + TESTING_HAPPEN_DIR)