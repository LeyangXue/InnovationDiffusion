# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:25 2022

@author: Leyang Xue
"""
root_path = 'F:/work/work4_dynamic' #please change the current path if run the code 

import sys
sys.path.append(root_path + '/InnovationDiffusion')
from utils import coupling_diffusion as cd
from multiprocessing import Pool 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random 
import pickle as pk

def RunSimuBetaC(args):
    '''
    run the simulation with parallel computation

    Parameters
    ----------
    args : list
        paramater sets.

    Returns
    -------
    r : float
        percentage of adopted individuals in the network.

    '''
    it_num,G,betac,mu,alpha,outbreak_origin = args
    print('simulation:%d'%it_num)

    Nnodes = G.order()
    
    [S,I,R] = cd.SIR_G_df(G,betac,mu,alpha,outbreak_origin)
    r = R[-1]/Nnodes
        
    return r

def RhoBetacSpread(G,betac_DMP,alphas,file_path):
    '''
    run the numerical simulation with alphas and a betac_dmp

    Parameters
    ----------
    G : graph
        network.
    betac_DMP : float
        critical point obtained from DMP.
    alphas : float
        intensity of comformity.
    file_path : str
        path to save the result.

    Returns
    -------
    None.

    '''
    simulation = 5000000
    mu = 1
    
    for alpha in alphas:
        #for a given alpha
        args = []
        for i in np.arange(simulation): 
            infected = random.choice(list(G.nodes()))
            args.append([i,G,betac_DMP,mu,alpha,infected])
    
        pool = Pool(processes=50)    
        results_mc = pool.map(RunSimuBetaC,args)
        pool.close()
        pk.dump(results_mc, open(file_path+'/'+str(alpha)+'_betac_mc.pkl', "wb"))
        
def PlotAxes(ax,xlabel,ylabel,title):
    
    fontsize = 13
    n_legend = 13
    
    font_label = {'family': "Arial", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    ax.set_title(title, loc='left',fontdict = {'family': "Arial", 'size':n_legend})
    ax.set_xticks([0.00001,0.001, 0.1])
    ax.set_yticks([0.0000001,0.00001,0.001,0.1])
    #ax.text(index[0],index[1],title,fontdict = font_label)
    
if __name__ == '__main__':   
    
    root_path = 'F:/work/work4_dynamic' #please change the current path if run the code 
    network_path = root_path + '/InnovationDiffusion/sfigure5/network'
    result_path =  root_path + '/InnovationDiffusion/sfigure5/result'
    spread_path =  root_path + '/InnovationDiffusion/sfigure5/spread'
    figure_path =  root_path + '/InnovationDiffusion/sfigure5/figure'
    
    #run the simulation for various numerical alpha_c 
    network = 'soc-fb-pages-artist.txt'
        
    #set the network name       
    networkname  = network.split('.txt')[0]
    
    #set the basic parameter 
    betac_dmp = 0.0057
    netname = 'soc-fb-pages-artist'
    alphas = list(map(lambda x: round(x,2), np.arange(0.12,0.20,0.01)))
    
    #create new files
    spreadpath = spread_path+'/'+networkname
    cd.mkdir(spreadpath)
    
    #load the network
    data = np.loadtxt(network_path+'/'+network)
    G = cd.load_network(data)
    
    #spreading on the network with a given betac and alpha
    RhoBetacSpread(G,betac_dmp,alphas,spreadpath)
    
    #numerical alpha_c identified by mass distribution of rhos
    #plot the result
    #file = '/Assort_p0.6_soc-fb-pages-artis.edgelist/'
    fontsize = 14
    font_label = {'family': "Arial", 'size':fontsize}
    titles = ['(a)', '(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    figure = "massdistribution"
    
    
    alphas = list(map(lambda x: round(x,2),np.arange(0.12,0.2,0.01)))
    mass = {}
    color = '#4063a3'
    n = 3
    row = int(np.ceil(len(alphas)/n))
    fig, ax = plt.subplots(row,n,figsize=(n*2.5,row*2.5), sharey=True, sharex=True, constrained_layout=True)
    for i,alpha in enumerate(alphas):
        x = int(i/n)
        y = int(i%n)
        rhos_mc = cd.load(spreadpath+'/'+str(alpha)+'_betac_mc.pkl')
        mass[alpha]= cd.massDistribution(rhos_mc)
        ax[x,y].loglog(mass[alpha].keys(),mass[alpha].values(),'o',mec=color,mfc='white',ms=6)
        ax[x,y].text(0.001,0.1,titles[i]+ r' $\alpha$='+str(alpha),fontdict=font_label)
        PlotAxes(ax[x,y],'','','')#titles[i]+ r'       $\alpha$='+str(alpha)
    
    PlotAxes(ax[0,0],'','  ', '')#r'(a) $\alpha$=0.12'
    PlotAxes(ax[1,0],'','  ', '')#r'(d) $\alpha$=0.15'
    PlotAxes(ax[2,0],'  ','  ', '')#r'(g) $\alpha$=0.18'
    PlotAxes(ax[2,1],'  ','','')#r'(h) $\alpha$=0.19'
    PlotAxes(ax[2,2],'  ','','')#r'(i) $\alpha$=0.20'
    
    fig.text(0.32,0.01,'Proportion of adopted clusters,  '+ '$m/N$', fontdict = font_label)
    fig.text(0.0,0.14,'Probability of observing clusters of adopters of different sizes, '+ '$P(m/N)$', fontdict = font_label, rotation='vertical')

    plt.savefig(figure_path+'/'+figure+'.png',dpi=300)
    plt.savefig(figure_path+'/'+figure+'.pdf')
    plt.savefig(figure_path+'/'+figure+'.eps')
