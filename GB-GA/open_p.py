import pickle
r = open('test.p','rb')
r = pickle.load(r)
print(r)