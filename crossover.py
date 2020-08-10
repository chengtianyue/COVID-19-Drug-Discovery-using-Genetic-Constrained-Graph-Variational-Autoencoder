import random
def gap(smls):
	#输出包含所有括号位置的列表
	parlocs = []
	ghalocs = []
	for a in range(len(smls)):
		if smls[a] == '(':
			ghalocs.insert(0,a)
		elif smls[a] == ')':
			parlocs.append([ghalocs[0],a])
			ghalocs.pop(0)
	return parlocs


def swapbranches(sms1,sms2,parlocs1,parlocs2):
	#随机交换两个分子的支链，输入为分子1、分子2、分子1括号位置、分子2括号位置
	locs1 = random.choice(parlocs1)
	locs2 = random.choice(parlocs2)
	tmp = sms1[locs1[0]:locs1[1]+1]
	sms1 = sms1[:locs1[0]] + sms2[locs2[0]:locs2[1]+1] + sms1[locs1[1]+1:]
	sms2 = sms2[:locs2[0]] + tmp + sms2[locs2[1]+1:]
	return sms1, sms2


def reproduce(smslist,parfinder,reproduction_metric=2, prob=1):
	for b in range(reproduction_metric):
		random.shuffle(smslist)
		for a in range(int(len(smslist)/2)):
			if prob > random.random():
				smslist += swapbranches(smslist[a*2],smslist[a*2+1],parfinder(smslist[a*2]),parfinder(smslist[a*2+1]))
			else:
				smslist += (smslist *2)
	#print(smslist, '=smslist', len(smslist))
	return smslist
