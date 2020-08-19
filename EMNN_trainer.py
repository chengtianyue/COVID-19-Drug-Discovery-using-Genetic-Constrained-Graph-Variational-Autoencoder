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
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import EMNN.gnn
import EMNN.gnn.emn_implementations
from EMNN.losses import LOSS_FUNCTIONS
from EMNN.train_logging import LOG_FUNCTIONS
from EMNN.gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn
from EMNN.train_logging import feed_net
from EMNN.train_logging import compute_mse
import datetime
from torch.utils.tensorboard import SummaryWriter    
print("import done")
#Train test split
train_dataset = MolGraphDataset('Data/protease_train.csv.gz')
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)
validation_dataset = MolGraphDataset('Data/protease_valid.csv.gz')
validation_dataloader = DataLoader(validation_dataset, batch_size=50, collate_fn=molgraph_collate_fn)
test_dataset = MolGraphDataset('Data/protease_test.csv.gz')
test_dataloader = DataLoader(test_dataset, batch_size=50, collate_fn=molgraph_collate_fn)
#Training Prams
((sample_adjacency, sample_nodes, sample_edges), sample_target) = train_dataset[0]

net = EMNN.gnn.emn_implementations.EMNImplementation(node_features=len(sample_nodes[0]), 
                                                edge_features=len(sample_edges[0, 0]), 
                                                out_features=len(sample_target), 
                                                message_passes=8, edge_embedding_size=50, 
                                                edge_emb_depth=2, edge_emb_hidden_dim=150, 
                                                edge_emb_dropout_p=0.0, att_depth=2, att_hidden_dim=85, 
                                                att_dropout_p=0.0, msg_depth=2, msg_hidden_dim=150, 
                                                msg_dropout_p=0.0, gather_width=45, gather_att_depth=2, 
                                                gather_att_hidden_dim=45, gather_att_dropout_p=0.0, 
                                                gather_emb_depth=2, gather_emb_hidden_dim=45, 
                                                gather_emb_dropout_p=0.0, out_depth=2, out_hidden_dim=450, 
                                                out_dropout_p=0.1, out_layer_shrinkage=0.6)
                                                
if True:
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.MSELoss()
writer = SummaryWriter('8_14_2020_1/')#Change this to current date
SAVEDMODELS_DIR = "EMNN/savedmodels/"
def evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion):
    global evaluate_called
    global DATETIME_STR
    global best_mean_train_score
    global best_mean_validation_score
    global best_mean_test_score
    global train_subset_loader
    
    if not evaluate_called:
        evaluate_called = True
        best_mean_train_score, best_mean_validation_score, best_mean_test_score = 10, 10, 10
        train_subset_loader = train_dataloader

    train_output, train_loss, train_target = feed_net(net, train_subset_loader, criterion, True)
    validation_output, validation_loss, validation_target = feed_net(net, validation_dataloader, criterion, True)
    test_output, test_loss, test_target = feed_net(net, test_dataloader, criterion, True)

    train_scores = compute_mse(train_output, train_target)
    train_mean_score = np.nanmean(train_scores)
    validation_scores = compute_mse(validation_output, validation_target)
    validation_mean_score = np.nanmean(validation_scores)
    test_scores = compute_mse(test_output, test_target)
    test_mean_score = np.nanmean(test_scores)

    new_best_model_found = validation_mean_score < best_mean_validation_score

    if new_best_model_found:
        best_mean_train_score = train_mean_score
        best_mean_validation_score = validation_mean_score
        best_mean_test_score = test_mean_score

        path = SAVEDMODELS_DIR + type(net).__name__ + DATETIME_STR
        torch.save(net, path)

    target_names = train_dataloader.dataset.target_names
    return {  # if made deeper, tensorboardx writing breaks I think
        'loss': {'train': train_loss, 'test': test_loss},
        'mean {}'.format("MSE"):
            {'train': train_mean_score, 'validation': validation_mean_score, 'test': test_mean_score},
        'train {}s'.format("MSE"): {target_names[i]: train_scores[i] for i in range(len(target_names))},
        'test {}s'.format("MSE"): {target_names[i]: test_scores[i] for i in range(len(target_names))},
        'best mean {}'.format("MSE"):
            {'train': best_mean_train_score, 'validation': best_mean_validation_score, 'test': best_mean_test_score}
    }
def less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch):
    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion)
    mean_score_key = 'mean {}'.format("MSE")
    writer.add_scalar('Train/Train_MSE', scalars[mean_score_key]['train'].item(), epoch)
    writer.add_scalar('Train/Validation_MSE', scalars[mean_score_key]['validation'].item(), epoch)
    writer.add_scalar('Train/Test_MSE', scalars[mean_score_key]['test'].item(), epoch)
    writer.flush()
    print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}: {}'.format(
        epoch + 1,
        "MSE", scalars[mean_score_key]['train'],
        "MSE", scalars[mean_score_key]['validation'],
        "MSE", scalars[mean_score_key]['test'])
    )
evaluate_called = False
best_mean_train_score, best_mean_validation_score, best_mean_test_score = 10, 10, 10
train_subset_loader = None
DATETIME_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

for epoch in range(50):
    net.train()
    for i_batch, batch in enumerate(train_dataloader):

        if True:
            batch = [tensor.cuda() for tensor in batch]
        adjacency, nodes, edges, target = batch

        optimizer.zero_grad()
        output = net(adjacency, nodes, edges)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
        optimizer.step()
    

    with torch.no_grad():
        net.eval()
        less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch)

print("Finished Training")
def predict(test_set):
    with torch.no_grad():
        #Change this path to predict using different trained models
        net = torch.load("EMNN/savedmodels/EMNImplementation2020-08-13 18:40:10.597721")#Change to new model
        if True:
            net = net.cuda()
        else:
            net = net.cpu()
        net.eval()

        dataset = MolGraphDataset(test_set, prediction=True)
        dataloader = DataLoader(dataset, batch_size=50, collate_fn=molgraph_collate_fn)

        batch_outputs = []
        for i_batch, batch in enumerate(dataloader):
            if True:
                batch = [tensor.cuda() for tensor in batch]
            adjacency, nodes, edges, target = batch
            batch_output = net(adjacency, nodes, edges)
            batch_outputs.append(batch_output)

        output = torch.cat(batch_outputs).cpu().numpy()
        
        df = pd.read_csv(test_set)
        
        df.insert(1,'pred_log_std_scaled', output[:,0], True)
        
        return df
predictions = predict("Data/n3_similarity_test.csv.gz")
predictions = predictions.sort_values("pred_log_std_scaled", ascending=False)
best_predicted = predictions[["empty\tsmiles"]].values[:10,0]
best_predicted = [best_predicted[i][1:] for i in range(len(best_predicted))]
pickle.dump(best_predicted, open("Data/best_predicted_smiles.pkl", "wb"))
