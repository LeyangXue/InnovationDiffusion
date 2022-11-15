# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 09:47:04 2022

@author: Leyang Xue
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def PlotAxes(ax,xlabel,ylabel,title):
    
    fontsize = 24
    n_legend = 22
    
    font_label = {'family': "Arial", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    ax.legend(loc='best',framealpha=0, fontsize=n_legend,handletextpad=0.1)
    ax.set_title(title, loc='left',fontdict = font_label)
    #ax.text(index[0],index[1],title,fontdict = font_label)
    
if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #please change the current path if run this code  
    result_path = root_path + '/InnovationDiffusion/sfigure4/result'
    figure_path = root_path + '/InnovationDiffusion/sfigure4/figure'

    #set the name of networks 
    realnetwork = ['soc-fb-pages-artist.csv','advogato.csv', 'InterdisPhysics.csv','soc-delicious.csv']
    realnames = ['Fb-pages-artist','Advogato','InterdisPhysics', 'Delicious']
    modelnetwork = ['22_statistic_result.csv','26_statistic_result.csv','33_statistic_result.csv']
    modelnames = ['gamma22','gamma26','gamma33']
    
    #load the realnetwork datasets
    realnetdata  = {}
    for name, file in zip(realnames, realnetwork):
        realnetdata[name] = pd.read_csv(result_path+'/realnetwork/'+file)
        
    #load the network model datasets
    modelnetdata = {}
    for name, file in zip(modelnames,modelnetwork):
        modelnetdata[name] = pd.read_csv(result_path+'/modelnetwork/'+file)
    
    #plot the results
    fontsize = 24
    ms=14
    font_label = {'family': "Arial", 'size':fontsize}
    colors = ["#ddc000","#79ad41","#34b6c6","#4063a3"]

    fig, ax = plt.subplots(1,2, figsize = (12,6), sharey=True,constrained_layout=True)
    #Fb-pages-artist
    y1 = realnetdata['Fb-pages-artist'].sort_values(by='LS', ascending = bool)['LS']
    x1 = list(np.arange(-10,0,2))
    x1.extend(list(np.arange(1,11,2)))
    ax[0].plot(x1,y1,'o-',color=colors[0], markersize = ms, mec='white', label='Fb-pages-artist') 
    
    #Advogato
    y2 = realnetdata['Advogato'].sort_values(by='LS', ascending = bool)['LS']
    x2 = list(np.arange(-10,0,1))
    x2.extend(list(np.arange(1,11,1)))
    ax[0].plot(x2,y2,'s-',color=colors[1], markersize = ms, mec='white', label='Advogato') 
    
    #InterdisPhysics
    y3 = realnetdata['InterdisPhysics'].sort_values(by='LS', ascending = bool)['LS']
    x3 = list(np.arange(-10,0,1))
    x3.extend(list(np.arange(1,11,1)))
    ax[0].plot(x3,y3,'^-',color=colors[2], markersize = ms, mec='white', label='InterdisPhysics') 
    
    #delicious
    y4 = realnetdata['Delicious'].sort_values(by='LS', ascending = bool)['LS']
    x4 = list(np.arange(-10,0,2))
    x4.extend(list(np.arange(2,11,2))) 
    ax[0].plot(x4,y4,'h-',color=colors[3], markersize = ms, mec='white', label='Delicious') 
    
    xticklabels = ['1.0','0.6','0.2','0.2','0.6','1.0']
    #ax[0].set_yscale('log')
    ax[0].set_xticks([-10,-6,-2,2,6,10])
    ax[0].set_xticklabels(xticklabels)
    PlotAxes(ax[0],'  ', r'Localization strength, $\mathcal{L}$','(a)')
    
    y22 = modelnetdata['gamma22'].sort_values(by='LS', ascending = bool)['LS']
    ax[1].plot(x2,y22,'o-',color=colors[0], markersize = ms, mec='white', label=r'$\gamma=2.2$')
    y26 = modelnetdata['gamma26'].sort_values(by='LS', ascending = bool)['LS']
    ax[1].plot(x2,y26,'s-',color=colors[1], markersize = ms, mec='white', label=r'$\gamma=2.6$')
    y33 = modelnetdata['gamma33'].sort_values(by='LS', ascending = bool)['LS']
    ax[1].plot(x2,y33,'^-',color=colors[2], markersize = ms, mec='white', label=r'$\gamma=3.3$')
    ax[1].set_xticks([-10,-6,-2,2,6,10])
    ax[1].set_xticklabels(xticklabels)
    ax[1].annotate("Disassort.", xy=(-10,0.055), xytext=(-7.5, 0.025),arrowprops=dict(arrowstyle="->"), size=fontsize)
    ax[1].annotate("Assort.", xy=(10, 0.055), xytext=(1.8,0.025),arrowprops=dict(arrowstyle="->"), size=fontsize)
  
    PlotAxes(ax[1],'  ', '','(b)')
    fig.text(0.3,0.02,'Proportion of rewired edges, p', fontdict=font_label)
    
    plt.savefig(figure_path+'/AssortLS.png',dpi=300)
    plt.savefig(figure_path+'/AssortLS.pdf')
    plt.savefig(figure_path+'/AssortLS.eps')

    