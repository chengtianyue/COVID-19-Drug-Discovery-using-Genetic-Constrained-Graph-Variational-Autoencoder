import random
from rdkit import rdBase, Chem, DataStructs
import numpy

def getflist(molist):
	#输出包含每个分子适应值的列表
	flist = []
	dcoy = molist
	for a in range(len(dcoy)):
		fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(dcoy[0]))
		mem = dcoy[0]
		del dcoy[0]
		similarity = np.mean(DataStructs.BulkTanimotoSimilarity(fingerprint,dcoy))
		dcoy.append(mem)
		flist.append(fingerprint)
	return flist

def select(molist,flist,reproduction_metric=2):
	#轮盘赌选择+精英保留
	#flist为含有每个样本适应度值的list，crat为希望选择进下一轮的样本比例
	comp = []
	for a in molist:
		comp.append([a])
	for a in range(len(flist)):
		comp[a].append(flist[a])
	chosen = []
	n = len(flist)
	for i in range(n-1):
		for j in range(i+1,n):
			if comp[i][1]<comp[j][1]: 
				comp[i],comp[j]=comp[j],comp[i]	
	lenf = len(flist) - 1
	cnumb = int(lenf / (2*(reproduction_metric**2))) + 1
	#进入下一轮的样本数量为1/(默认4)
	for a in range(cnumb):
		chosen.append(comp[0][0])
		del comp[0]
	#选择适应值最高的一部分样本进入下一轮
	for a in range(cnumb):
		chosen.append(random.choice(comp)[0])
	#再随机选择一部分样本进入下一轮
	molist = chosen
	return chosen
