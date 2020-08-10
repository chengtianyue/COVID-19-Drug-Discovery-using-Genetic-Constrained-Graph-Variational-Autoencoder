import numpy as np
import random

from parts.number_system_convertor import trans
    #用于进制转换，输入和输出均为list形式（list中每个数字分别代表数字的一位）。示例：nsc.trans(旧进制，新进制，旧进制中的数字)
from parts.crossover import gap
#输出包含所有括号位置的列表
from parts.parenshifter import gpp
#输出包含所有最内层括号位置的列表
from parts.selection import getflist
	#输入所有分子SMILES以及打分函数，输出
from parts.selection import select
    #轮盘赌选择+精英保留
    #flist为含有每个样本适应度值的list
from parts.selection import score
	#给分子评分的机制
from parts.crossover import reproduce
	#增加样本量并随机crossover，输入为包含多个SMILES表达式的列表以及获取括号位置的函数以及繁殖系数（后者可不输入，默认为2），输出为包含reproduce后SMILES表达式的列表。
from Topping import cut
#输入大列表、希望输出的元素数目、希望输出的元素组号，输出大列表的【自定义】个元素。组号从0开始。
import pickle
with open('smiles_zinc.pkl', "rb+") as smiles:
	Tsmslist = pickle.load(smiles)
#print(len(Tsmslist))
#print(Tsmslist)
#至此初始化完成

def GA(population,scorer,generations):
	for a in population:
		print(a)
	print('='*150)
	for a in range(generations):
		reproduce(population,gpp)
		reproduce(population,gap)
		for b in range(2):
			flist = getflist(population,scorer)
			population = select(population,flist)
		print(a,len(population))
		print(population)
	return population

GA(cut(Tsmslist,100,0),score,10)
