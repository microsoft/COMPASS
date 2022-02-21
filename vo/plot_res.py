# plot two rewards from logging files for comparison
import matplotlib as mpl
from cycler import cycler
import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt

# set the color cycler for the plotting
colors= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bb3f3f', 'g','b','#c8fd3d','m','k','#0a481e','#db4bda']
mpl.rcParams['axes.prop_cycle'] = cycler('color',colors)
mpl.rcParams.update({'font.size': 18})

# rewardlen = 3000
filedir = 'logdata'

prefs = ['11_0_', '11_3_', '11_4_', '11_1_', '11_2_']
plots = ['test.npy'] # , 'test.npy', 'loss.npy'
labs = ['Testing Loss'] # ,'Testing Loss', 'Training Loss'
PlotLen=100000
files = []
labels = []
for pref in prefs:
    for plot,lab in zip(plots,labs):
        files.append(pref + plot)
        labels.append(pref[:-1]+' '+lab)
# files = [pref+fn for fn in ['loss.npy', 'test.npy']]#['trans_loss.npy', 'test_trans_loss.npy', 'rot_loss.npy', 'test_rot_loss.npy']] 
# labels = ['Training Trans Loss','Testing Trans Loss', 'Training Rot Loss','Testing Rot Loss']

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    
    flagR = False
    if len(datax)%group>0:
        rx = datax[ -(len(datax) % group):].max()
        ry = datay[ -(len(datay) % group):].mean()

        rx = np.array([rx])
        ry = np.array([ry])
        
        flagR = True
        
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    
    if flagR:
        datax = np.concatenate((datax, rx), axis=0)
        datay = np.concatenate((datay, ry), axis=0)
    
    return (datax, datay)


plt.figure(figsize=(5,6))
for k,filename in enumerate(files):
    filepathname = join(filedir, filename)
    reward_acc = np.load(filepathname)
    # import ipdb;ipdb.set_trace()
    reward_acc = reward_acc[reward_acc[:,0]<=PlotLen,:]
    print 'load:', filename,reward_acc.shape
    if True: #k==0 or k==2:
        plt.plot(reward_acc[:,0],reward_acc[:,1], alpha=0.2,color=colors[k])
        gx, gy = groupPlot(reward_acc[:,0], reward_acc[:,1], group=20)
        plt.plot(gx,gy,label=labels[k],color=colors[k]) #gx,
    else:
        plt.plot(reward_acc[:,0],reward_acc[:,1],label=labels[k],color=colors[k])
plt.legend(prop={'size': 18}, loc='upper right')
plt.xlabel('Number of iterations (k)')
plt.ylabel('Loss')
# plt.legend([filename.split('_')[0]+'_'+filename.split('_')[1] for filename in files],prop={'size': 13})
# plt.legend(['RL with option','RL with option avg', 'flat RL','flat RL avg'],prop={'size': 13})
# plt.xlim(0,2000)
plt.ylim(0,0.7)
plt.grid()
plt.show()