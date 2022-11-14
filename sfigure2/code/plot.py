# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:45:40 2022

@author: Leyang Xue
"""

import os 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def PlotAxes(ax,xlabel,ylabel, title, mode=False):
    
    fontsize = 28
    font_label = {'family': "Arial", 'size':fontsize}
    
    n_legend = 25
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = font_label,pad=10)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    if mode == True:
        ax.legend(loc='upper left',bbox_to_anchor=(0.00,1.0), framealpha=0, fontsize=30)
            
def NetworkSort(properities,netsname):
    
    statistic = pd.read_csv(result_path+'/complete_datasets.csv')
    sortofnetwork  = statistic.sort_values(by=properities) #rank the network according to the size of network
    netsname_rank = [net for net in sortofnetwork['name'] if net in netsname]
    
    return netsname_rank

def loadSpreadData(netsortsize):
    
    results = pd.read_csv(result_path+'/summary_results.csv')
    index = list(map(lambda x:round(x,2),np.arange(0,1.01,0.05)))
    betac_name = pd.DataFrame(index=index,columns=netsortsize)
    for name in netsortsize:
        result = results[results['network']==name]
        betac_name[name] = list(result['betac_mc'] - result['betac_dmp'])
    
    return betac_name

def plotviolinplot(betacdata,networklabel,figurename):
    
    fontsize = 28
    font_label = {'family': "Arial", 'size':fontsize}
    #color = ["#4063a3",'#cfcfcf','#ddc000','#d7aca1',"#34b6c6","#79ad41",'#3b3a3e']
    
    fig, ax = plt.subplots(1,1,figsize=(16,8), constrained_layout = True)
    vp = ax.violinplot(betacdata, np.arange(0,len(netsname),1),widths=0.6,showmeans=False, showmedians=False, showextrema=True)
    ax.set_ylim(-0.2,0.2)
    ax.set_xlim(-0.5,37.5)
    ax.set_xticks(np.arange(0,len(netsname),1))
    ax.set_xticklabels(networklabel, rotation=90)
    ax.grid(axis='both')
    PlotAxes(ax,'Networks',' \n', '')
    fig.text(0.01,0.35,'Numerical '+ r'$\beta_c$ ' +'- Theoretical '+ r'$\beta_c$',rotation=90, fontdict=font_label)
    # styling:
    for body in vp['bodies']:
        body.set_alpha(0.35)
        #body.set_facecolor(color[4])
        body.set_edgecolor('white')   
        
    plt.savefig(figure_path+'/'+ figurename + '.png', dpi=300)
    plt.savefig(figure_path+'/'+ figurename + '.eps')
    plt.savefig(figure_path+'/'+ figurename + '.pdf')
    
if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #need to change if run this code
    result_path = root_path + '/InnovationDiffusion/sfigure2/result'
    network_path =  root_path + '/InnovationDiffusion/sfigure2/network'
    figure_path = root_path + '/InnovationDiffusion/sfigure2/figure'
    
    #set the label
    netsname = [each.split('.txt')[0] for each in sorted(os.listdir(network_path))]
    
    #rank the network according to the size of networks    
    netsortsize= NetworkSort('N',netsname)
    #rank the network according to the mean degree of network
    netsortdegree= NetworkSort('k',netsname)

    #load the data
    betac_sortsize = loadSpreadData(netsortsize)
    betac_sortdegree = loadSpreadData(netsortdegree)

    #plot the figure
    plotviolinplot(betac_sortsize,netsortsize,'sfigure2_ranksize')
    plotviolinplot(betac_sortdegree,netsortdegree,'sfigure2_rankdegree')
    