f1 = open('testfolder2.txt')
f2 = open('set1list.txt')

f1list = []
f2list = []
scorelist = []
truthlist = []
for i in f1:
	f1list.append(i.replace('\n','').split('\t')[0])
	scorelist.append(float(i.replace('\n','').split('\t')[1]))

for i in f2:
	aa = i.replace('\n','').split('\\')[-1].replace('.jpg','')
	f2list.append(aa)

assert len(f1list)==len(f2list)

cnt = 0
for i in range(len(f1list)):
	if f1list[i]==f2list[i]:
		truthlist.append(1)
	else:
		truthlist.append(0)

import matplotlib.pyplot as plt 
import numpy as np 
total = list(zip(truthlist,scorelist))
srt = sorted(total,key=lambda x: x[1],reverse=True)
# print(srt)
laa = []
truenumber = 0
for i in range(500):
	# cnt += srt[i][0]
	truenumber += srt[i][0]
	laa.append(float(truenumber)/(i+1))

plt.ion()
plt.plot(np.array(list(range(len(laa))))/500,laa)
plt.ylim(0.95,1)
# plt.xlim(0,0.05)
plt.show()
plt.grid(True)
input()

print('correct:',cnt)