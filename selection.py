import random
def select(flist, crat=0.25):
	#轮盘赌选择+精英保留
	#flist为含有每个样本适应度值的list，crat为希望选择进下一轮的样本比例
	chosen = []
	flist.sort()
	lenf = len(flist)-1
	cnumb = int(lenf * crat/2)
	#进入下一轮的样本数量的1/2
	for a in range(cnumb):
		chosen.append(flist[lenf-a])
		del flist[-1]
	#选择适应值最高的一部分样本进入下一轮
	for a in range(cnumb):
		chosen.append(random.choice(flist))
	#再随机选择一部分样本进入下一轮
	print(chosen)
	return chosen



#select(f1)