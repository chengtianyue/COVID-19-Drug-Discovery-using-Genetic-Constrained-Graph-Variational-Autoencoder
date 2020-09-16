import pickle
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
from rdkit.six import StringIO
# import_path = '110epoch_generated_smiles_zinc'
# print("Finished Extracting SMILES from "+import_path)
# with open(import_path, 'rb') as f:
#     x = pickle.load(f)
x = [
    'CC[C@@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)N',
'CC[C@@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)N',
'CC[C@@H](c1ccc(c(c1F)C(=O)c1ccc(nc1)N)Cl)N',
'N#Cc1ccc(cc1)C(=O)c1c(Cl)ccc(c1F)[C@H](C1CC1)N',
'OCCN[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)CC',
'CC[C@@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)NCCC(=O)N',
'OCCCN[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)CC',
'CCC(c1ccc(c(c1F)C(=O)c1ccccc1)Cl)NC(CO)(C)C',
'NCCC(=O)N[C@H](c1ccc(c(c1F)C(=O)c1ccc(nc1)N)Cl)CC',
'CC[C@@H](c1ccc(c(c1F)C(=O)c1cccnc1)Cl)NC(CC(=O)N)(C)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)F)N[C@H](CC(=O)N)C',
'CC[C@@H](c1ccc(c(c1F)C(=O)c1ccc(nc1)N)Cl)NC(CC(=O)N)(C)C',
'C[C@H](N[C@@H](c1ccc(c(c1F)C(=O)c1ccncc1)Cl)C1CC1)CC(=O)N',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccc(nc1)N)Cl)N[C@H](CO)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)C)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccc(cc1)F)Cl)N[C@H](CC(=O)N)C',
'C[C@H](N[C@@H](c1ccc(c(c1F)C(=O)c1ccccc1)Cl)C1CC1)CC(=O)N',
'CC[C@H](c1ccc(c(c1F)C(=O)c1cccc(c1)F)Cl)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccccc1)OC)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccns1)Cl)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1cccnc1)Cl)NC(=O)C[C@@H](C(C)C)N',
'CC[C@H](c1ccc(c(c1F)C(=O)c1ccc(cc1)F)OC)N[C@H](CC(=O)N)C',
'COc1ccc(c(c1C(=O)c1ccc(cc1)F)F)[C@@H](C1CC1)N[C@H](CC(=O)N)C',
'CC[C@H](c1ccc(c(c1F)C(=O)c1cccc(c1)F)OC)N[C@H](CC(=O)N)C',
'NCCNC(=O)C[C@@H](N[C@H](c1ccc(c(c1F)C(=O)c1ccc(nc1)N)Cl)CC)C'

]
mols = []
results = [] 
print("Saving SDFs at Folder 'SDFs'")
for i in range(len(x)):
    results.append(Chem.MolFromSmiles(x[i]))
    sio = StringIO()
    w = Chem.SDWriter('SD/%s.sdf'%str(i))
    for m in mols: w.write(m)
    w.flush()
print("Finished converting")