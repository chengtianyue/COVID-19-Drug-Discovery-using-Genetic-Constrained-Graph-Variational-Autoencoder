import pickle
r = open('generated_smiles_zinc','rb')
r = pickle.load(r)
b = open('smiles_new.smi','w+')
for i in range(0,100):
    b.write(r[i])
    b.write('\n')
b.close()