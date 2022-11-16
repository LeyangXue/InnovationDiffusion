# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:13:24 2022

@author: Leyang Xue
"""

root_path = 'F:/work/work4_dynamic' #need to change if run this code

import sys
sys.path.append(root_path+'/InnovationDiffusion')
from utils import coupling_diffusion as cd
from multiprocessing import Pool 
import networkx as nx
import numpy as np
import random 
import pickle as pk
import pandas as pd

def RunOneSimulation(args):
    '''
    This function could be run with paralell computing
    
    Parameters
    ----------
    args : list
        parameter sets.

    Returns
    -------
    r_alphas : list
        infected ratio.
    '''
    
    it_num,G,betas,mu,alpha,outbreak_origin = args
    Nnodes = G.order()
    #print('simulation:%d'%it_num)
    
    r_alphas_df = []
    for j, beta in enumerate([betas]):
        [s,i,r] = cd.SIR_G_df(G,beta,mu,alpha,outbreak_origin)
        r_alphas_df.append(r[-1]/Nnodes)
        
    return r_alphas_df

def ParseResult(results,save_path,resultname):
    '''
    Parse the result obtained from the paralell computing
    
    Parameters
    ----------
    results : (array,array)
        spreading result
    save_path : str
        the path to save the result.
    file : str
        file name .

    Returns
    -------
    spreading : array
        SIR spreading result
    spreading_df : array
        SIR with frequence-dependent effect
    '''
    rhos_df_list = []
    for rhos_df in results:
        rhos_df_list.append(rhos_df)
        
    spreading_df = np.array(rhos_df_list)
    np.savetxt(save_path+'/'+resultname+'.csv',spreading_df, delimiter = ',')
    
    return spreading_df

def NetSpreadMC(G,networkName,simutimes,ResultPath,alpha,betas):
    '''
    identifying the betac on a given alpha through numerious simulation 

    Parameters
    ----------
    G : graph
        network.
    networkName : str
        network name.
    simutimes : int
        simulation times.
    ResultPath : str
        path to save the result.
    alpha : float
        intensity of conformity.
    betas : array
        adoption probability.

    Returns
    -------
    betac_mc : float
        numerical critical point.

    '''
    #set the spreading parameter
    mu = 1
    
    args = []    
    for it_num in np.arange(simutimes):
        outbreak_origin = random.choice(list(G.nodes()))    
        args.append([it_num,G,betas,mu,alpha,outbreak_origin]) 

    #1. run the mc with paralell computing 
    pool = Pool(processes=30)    
    results_mc = pool.map(RunOneSimulation,args)
    pool.close()
    resultname = networkName + '_results_mc_'+str(alpha)
    pk.dump(results_mc, open(ResultPath + '/'+resultname+'.pkl', "wb"))
    
    #identify the betac 
    spread = ParseResult(results_mc,ResultPath,resultname)
    index = cd.IdentifySimulationBetac(spread)
    betac_mc = betas[index]
    
    return betac_mc


if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #need to change if run this code
    network_path =  root_path + '/InnovationDiffusion/sfigure6/network'
    result_path = root_path + '/InnovationDiffusion/sfigure6/result'
    mcspread_path = root_path +'/InnovationDiffusion/sfigure6/mcspread'
    
    #networks 
    networks = ['soc-delicious.txt', 'soc-fb-pages-artist.txt','soc-advogato.txt','ca-InterdisPhysics.txt']
    #run the simulation with a given network 
    network = networks[0]
    #set the network name       
    networkname  = network.split('.txt')[0]
    #load the network inforamtion for each datatset
    data = pd.read_csv(result_path+'/'+networkname+'.csv')
    
    #path to save the spreading result from each dataset
    mcspreadpath =  mcspread_path + '/' +networkname
    cd.mkdir(mcspreadpath)

    #set the simulation times and alphas
    simutimes = 1000
    alpha = 1
    
    #create the dict to save the result
    netsname = sorted(data['network'])
    betac = {'network':[],'betac_dmp':[],'betac_mc':[]} #save the numerical critical point 
    betas_dict = {} #save the candidate range of beta
    for i, netname in enumerate(netsname):
        #each network 
        
        #load the networks
        print('i:',i,'netname:', netname)
        G = nx.read_edgelist(network_path+'/'+networkname+'/'+netname)
        
        #set the parameter in spreading 
        filename = netname.split('.edgelist')[0]
        betacdmp = float(data.loc[data['network']==netname]['DMP_betac'])
        intervals = np.array(list(map(lambda x: round(x,3),np.arange(-0.3,0.3,0.01))))
        betas = betacdmp + intervals
        betasnz = betas[betas>0]
        betas_dict[filename] = betasnz
        
        #calculate the numerical critical point with a given alpha, i.e. betac_mc
        betacmc= NetSpreadMC(G,filename,simutimes,mcspreadpath,alpha,betasnz)
        betac['network'].append(filename)
        betac['betac_dmp'].append(betacdmp)
        betac['betac_mc'].append(betacmc)
        
    #save the result from each networks
    betac_pd = pd.DataFrame(betac)
    betas_pd = pd.Series(betas)
    betac_pd.to_csv(mcspreadpath+'/' + networkname + '_betac.csv')    
    betas_pd.to_csv(mcspreadpath+'/' + networkname + '_betas.csv')