
import sys
import os

GIT_ROOT_DIR = "/home/levon003/repos/ALSim"  # hacky; hardcoding to levon003's execution environment
SRC_DIR = os.path.join(GIT_ROOT_DIR, 'src')

DATA_DIR = os.path.join(GIT_ROOT_DIR, 'data')
DERIVED_DATA_DIR = os.path.join(DATA_DIR, 'derived')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

def add_utils_to_path():
    sys.path.append(SRC_DIR)
