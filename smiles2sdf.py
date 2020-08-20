import pickle
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
from rdkit.six import StringIO
import_path = 'CGVAE/generated_smiles_zinc'
print("Finished Extracting SMILES from "+import_path)
with open(import_path, 'rb') as f:
    x = pickle.load(f)
mols = [] 
print("Saving SDFs at Folder 'SDFs'")
for i in range(len(x)):
    results.append(Chem.MolFromSmiles(x[i]))
    sio = StringIO()
    w = Chem.SDWriter('SDFs/%s.sdf'%str(i))
    for m in mols: w.write(m)
    w.flush()
print("Finished converting")