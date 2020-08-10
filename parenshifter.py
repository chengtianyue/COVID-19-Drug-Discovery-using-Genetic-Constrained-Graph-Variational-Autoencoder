def gpp(smls):
	#输出包含所有最内层括号位置的列表
	parlocs = []
	lpos1 = 0
	stat = False
	for a in range(len(smls)):
		if smls[a] == '(':
			lpos = a
			stat = True
		if smls[a] == ')':
			if stat == True:
				parlocs.append([lpos,a])
				stat = False
	return parlocs


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
