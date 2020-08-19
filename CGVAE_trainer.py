import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import cDataStructs
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import os
import time
import pickle
import csv
from rdkit.Chem import QED
import random
import pickle
import json
from sys import exit
import sys
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from typing import Sequence, Any
from docopt import docopt
from collections import defaultdict, deque
import sys, traceback
import pdb
from CGVAE.CGVAE import DenseGGNNChemModel
from CGVAE.GGNN_core import ChemModel
import CGVAE.utils
from CGVAE.utils import *
from numpy import linalg as LA
from copy import deepcopy
from CGVAE.data_augmentation import *

def train_valid_split(download_path):
    # load validation dataset
    with open("Data/valid_idx_zinc.json", 'r') as f:
        valid_idx = json.load(f)

    print('reading data...')
    raw_data = {'train': [], 'valid': []} # save the train, valid dataset.
    with open(download_path, 'r') as f:
        all_data = list(csv.DictReader(f))

    file_count=0
    for i, data_item in enumerate(all_data):
        smiles = data_item['smiles'].strip()
        QED = float(data_item['qed'])
        if i not in valid_idx:
            raw_data['train'].append({'smiles': smiles, 'QED': QED})
        else:
            raw_data['valid'].append({'smiles': smiles, 'QED': QED})
        file_count += 1
        if file_count % 2000 ==0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data

def preprocess(raw_data, dataset):
    print('parsing smiles as graphs...')
    processed_data = {'train': [], 'valid': []}
    
    file_count = 0
    for section in ['train', 'valid']:
        all_smiles = [] # record all smiles in training dataset
        for i,(smiles, QED) in enumerate([(mol['smiles'], mol['QED']) 
                                          for mol in raw_data[section]]):
            nodes, edges = to_graph(smiles, dataset)
            if len(edges) <= 0:
                continue
            processed_data[section].append({
                'targets': [[(QED)]],
                'graph': edges,
                'node_features': nodes,
                'smiles': smiles
            })
            all_smiles.append(smiles)
            if file_count % 2000 == 0:
                print('finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%      ' % (section))
        # save the dataset
        with open('Data/molecules_%s_%s.json' % (section, dataset), 'w') as f:
            json.dump(processed_data[section], f)
        # save all molecules in the training dataset
        if section == 'train':
            CGVAE.utils.dump('Data/smiles_%s.pkl' % dataset, all_smiles)  

df = pd.read_csv('Data/250k_rndm_zinc_drugs_clean_3.csv')
path = 'Data/250k_rndm_zinc_drugs_clean_3.csv'
raw_data = train_valid_split(path)
preprocess(raw_data, 'zinc')
args = {'--config': None,
 '--config-file': None,
 '--data_dir': 'Data/',
 '--dataset': 'zinc',
 '--freeze-graph-model': False,
 '--help': False,
 '--log_dir': 'CGVAE/',
 '--restore': 'CGVAE/10_zinc.pickle'}
model = DenseGGNNChemModel(args)
model.train()
