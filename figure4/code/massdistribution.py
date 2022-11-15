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
    simulation = 100000
    mu = 1
    
    for alpha in alphas:
        #for a given alpha
        args = []
        for i in np.arange(simulation): 
            infected = random.choice(list(G.nodes()))
            args.append([i,G,betac_DMP,mu,alpha,infected])
    
        pool = Pool(processes=40)    
        results_mc = pool.map(RunSimuBetaC,args)
        pool.close()
        pk.dump(results_mc, open(file_path+'/'+str(alpha)+'_betac_mc.pkl', "wb"))
        
    
if __name__ == '__main__':   
    
    root_path = 'F:/work/work4_dynamic' #please change the current path if run the code 
    network_path = root_path + '/InnovationDiffusion/figure4/network'
    result_path =  root_path + '/InnovationDiffusion/figure4/result'
    spread_path =  root_path + '/InnovationDiffusion/figure4/spread'
    
    #networks 
    networks = ['soc-delicious.txt', 'soc-fb-pages-artist.txt','soc-advogato.txt','ca-InterdisPhysics.txt']
    
    #run the simulation for various numerical alpha_c 
    for network in networks:
        
        #set the network name       
        networkname  = network.split('.txt')[0]
        
        #load the network inforamtion for each datatset
        data = pd.read_csv(result_path+'/'+networkname+'.csv')
        datasort = data.sort_values(by=['DMP_betac'])
    
        #for each network 
        for index in datasort.index:
           #set the basic parameter 
           alphac_p = datasort.iloc[index]['alphac_p']
           netname = datasort.iloc[index]['network']
           betac_dmp = datasort.iloc[index]['DMP_betac']
           
           #set the alphac, usually artificial
           alphac_standard= round(alphac_p,2)
           interval = list(map(lambda x: round(x,2),np.arange(-1,1,0.01))) #a example, need to change according to the experiment results 
           alphac_variation = alphac_standard + interval
           alphas = np.array([each for each in alphac_variation if each > 0]) 
           
           #create new files
           spreadpath = spread_path+'/'+networkname + '/'+netname
           cd.mkdir(spreadpath)

           #load the network
           G = nx.read_edgelist(result_path+'/'+networkname+'/'+netname)
           
           #spreading on the network with a given betac and alpha
           RhoBetacSpread(G,betac_dmp,alphas,spreadpath)
    
    #numerical alpha_c identified by mass distribution of rhos
    #plot the result
    file = '/Assort_p1.0_soc-deliciou/'
    figure = "fig4.png"
    networkname = networks[0]
    alphas = list(map(lambda x: round(x,2),np.arange(1.70,1.80,0.02)))
    mass = {}
    color = plt.get_cmap('Paired')
    n = 4
    row = int(np.ceil(len(alphas)/n))
    fig, ax = plt.subplots(row,n,figsize=(n*2,row*2), sharey=True, sharex=True, constrained_layout=True)
    for i,alpha in enumerate(alphas):
        x = int(i/n)
        y = int(i%n)
        rhos_mc = cd.load(spread_path+file+str(alpha)+'_betac_mc.pkl')
        mass[alpha]= cd.massDistribution(rhos_mc)
        ax[x,y].loglog(mass[alpha].keys(),mass[alpha].values(),'o',mec=color(1),mfc='white')
        ax[x,y].set_title(r'$\alpha$='+str(alpha))
        
    plt.savefig(spread_path+'/'+networkname+file+figure,dpi=200)
