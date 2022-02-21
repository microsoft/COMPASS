import numpy as np
# mm=np.load('kitti.npy')
# ll=[]
# for k in range(11):
#     with open('kitti_'+str(k).zfill(2)+'.txt','r') as f:
#         lines=f.readlines()
#         [ind,num]=lines[0].strip().split(' ')
#         ll.append(int(num))
# ll=np.array(ll)
# lll=ll-1
# mm1=mm[:np.sum(lll[:9]),:]
# mm2=mm[np.sum(lll[:9]):,:]
# np.save('kitti1.npy',mm1)
# np.save('kitti_test.npy',mm2)

# # ===== Split the tartan test set =====
# inputtxt = '../../geometry_vision/data/tartan_test.txt'
# inputnpy = '../../geometry_vision/data/tartan_test.npy'

# motions = np.load(inputnpy)

# txt1 = 'data/tartan_test.txt'
# txt2 = 'data/tartan_test_test.txt'

# with open(inputtxt,'r') as f:
# 	ll0 = f.readlines()
# 	ll0 = [ll.strip() for ll in ll0 if (not ll.startswith('00') and len(ll.strip())>0)]

# with open(txt1,'r') as f:
# 	ll1 = f.readlines()
# 	ll1 = [ll.strip() for ll in ll1 if (not ll.startswith('00') and len(ll.strip())>0)]

# with open(txt2,'r') as f:
# 	ll2 = f.readlines()
# 	ll2 = [ll.strip() for ll in ll2 if (not ll.startswith('00') and len(ll.strip())>0)]

# motions1=[]
# motions2=[]
# # import ipdb;ipdb.set_trace()
# # assert len(ll0)==len(ll1)+len(ll2)
# ind0,ind1,ind2 = 0,0,0
# motionind = 0
# for ind0 in range(len(ll0)):
# 	nums = ll0[ind0].split(' ')
# 	print nums
# 	num = int(nums[1])
# 	if ind1<len(ll1) and ll0[ind0]==ll1[ind1]:
# 		ind1 +=1
# 		motions1.extend(motions[motionind:motionind+num])
# 	elif( ind2<len(ll2) and ll0[ind0]==ll2[ind2]):
# 		ind2+=1
# 		motions2.extend(motions[motionind:motionind+num])
# 	else:
# 		print('!! Not find {}, {}'.format(ind0, ll0[ind0]))
# 	motionind += num
# import ipdb;ipdb.set_trace()
# np.save('tartan_test.npy',np.array(motions1))
# np.save('tartan_test_test.npy',np.array(motions2))

# ====== generate tartan test datafiles ========
from os import listdir, system
inputdir = '/bigdata/tartantest/hard_testing'
trajs=["MH00","MH01","MH02","MH03","MH04","MH05","MH06","MH07","SH00","SH01","SH02","SH03","SH04","SH05","SH06","SH07"]

for traj in trajs:
	datafile = 'data/tartantest/tartan_'+traj+'.txt'
	f = open(datafile, 'w')
	imgs = listdir(inputdir+'/'+traj+'/image_left')
	imgs = [ii for ii in imgs if ii.endswith('.png')]
	imgnum = len(imgs)
	print('{}, {}'.format(datafile,imgnum))
	f.write(traj + ' ' + str(imgnum-1) + '\n')
	for k in range(imgnum-1):
		f.write(str(k).zfill(6) + '\n')
	f.close()
	system('cp '+inputdir + '/' + traj + '/motion.npy data/tartantest/tartan_'+ traj + '.npy')
