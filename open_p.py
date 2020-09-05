import pickle
r = open('test.p','rb')
r = pickle.load(r)
a = r[199]
s = sorted(a,key = lambda x:x[0])
a = open('results.txt','w+')
for i in s:
    a.write(str(i[1]))
    a.write('\n')
a.close()
