# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:36:29 2021

@author: Leyang Xue
"""

#need to change the root path if run 
root_path = 'F:/work/work4_dynamic'

import sys
sys.path.append(root_path+'/InnovationDiffusion')
from utils import coupling_diffusion as cd
from multiprocessing import Pool 
import networkx as nx
import numpy as np
import random 
import pickle as pk

def NetNameMap(G):
    '''
    Map the node label of network

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    G_updated : graph
        network with updated label of node.

    '''
    nodedict ={node:i for i,node in enumerate(G.nodes())}
    L = []
    for edge in G.edges():
        L.append((nodedict[edge[0]],nodedict[edge[1]]))
    G_updated = nx.from_edgelist(L)
    
    return G_updated

def DMP(args):
    '''
    This function could be run with paralell computing
    
    Parameters
    ----------
    args : list
        parameter sets.

    Returns
    -------
    r_t : array
        DMP result with different alpha.

    '''
    it_num,G,alphas,mu,eta,outbreak_origin,Tmax = args
    Nnodes = G.order()
    print('simulation:%d'%it_num)
    
    r_t = np.zeros(len(alphas))
    directed = False
    edgelist = np.array([[int(each[0]),int(each[1])] for each in nx.to_edgelist(G)])
    for i,alpha in enumerate(alphas):
        output = cd.DMP_feedback(alpha,mu,eta,edgelist,int(outbreak_origin),Tmax,directed)
        s_time, i_time, r_time = output
        r_t[i] = np.sum(r_time, axis = 0)[-1]/Nnodes
    
    return r_t

def ParseResultDmp(results):
    '''
    Parse the DMP result

    Parameters
    ----------
    results : list
        DMP result.

    Returns
    -------
    rhos_array : array
        DMP with array.

    '''
    
    rhos = []
    for each in results:
         rhos.append(each)
         
    rhos_array  = np.array(rhos)
    
    return rhos_array

def NetSpreadDMP(networks,net_names,simutimes,Ndmp,ResultPath,Tmax,eta):
    '''
    run DMP method to calculation the result on network 

    Parameters
    ----------
    networks : graph
        network.
    net_names : str
        network filename.
    simutimes : int
        total numerical simulation times.
    Ndmp : int
        actual numerical simulation times.
    ResultPath : str
        the path to save the result.
    Tmax : int 
        step length of DMP.
    eta : float
        intensity of conformity.

    Returns
    -------
    None.

    '''
    #set the parameter
    betas = np.arange(0,1.01,0.01)
    mu = 1
        
    #construct the array of parameter to run simulation with multithreading  
    args_dmp = []
    for it_num in np.arange(simutimes):
        outbreak_origin = random.choice(list(networks.nodes()))
        args_dmp.append([it_num,networks,betas,mu,eta,outbreak_origin,Tmax])

    #run the DMP with paralell computing 
    argsDMP = random.sample(args_dmp,Ndmp)
    pool = Pool(processes=10)  
    results_dmp = pool.map(DMP,argsDMP)
    
    netname = net_names.split('.')[0]
    pk.dump(results_dmp, open(ResultPath + '/'+netname + '_results_dmp_'+str(eta)+'.pkl', "wb"))
    rhos_dmp = ParseResultDmp(results_dmp)
    np.savetxt(ResultPath+'/'+netname + '_dmp_'+str(eta)+'.csv', rhos_dmp, delimiter = ',', fmt = '%.4f')

def RunOneSimulation(args):
    '''
    This function could be run with paralell computing
    
    Parameters
    ----------
    args : list
        parameter sets.

    Returns
    -------
    r_alphas_df : list
        the percentage of adopted individuals.
    '''
    
    it_num,G,betas,mu,eta,outbreak_origin = args
    Nnodes = G.order()
    print('simulation:%d'%it_num)
    
    r_alphas_df = []
    for j, beta in enumerate([betas]):
        [s,i,r] = cd.SIR_G_df(G,beta,mu,eta,outbreak_origin)
        r_alphas_df.append(r[-1]/Nnodes)
        
    return r_alphas_df

def ParseResult(results,save_path,file,eta):
    '''
    Parse the numerical simulation
    
    Parameters
    ----------
    results : list
        spreading result
    save_path : str
        the path to save the result.
    file : str
        filename to save the numerical result with .csv format.

      Returns
    -------
    None.
    '''
    
    rhos_df_list = []
    for rhos_df in results:
        rhos_df_list.append(rhos_df)
        
    spreading_df = np.array(rhos_df_list)
    np.savetxt(save_path+'/'+file+ '_mc_'+str(eta)+'.csv',spreading_df, delimiter = ',')
    
def NetSpreadMC(G,networkName,simutimes,ResultPath,eta):
    
    ''' 
    run numerical simulation to calculation the result on network 
    
    Parameters
    ----------
    G : networkx graph
        network.
    networkName : str
        network filename.
    simutimes : int
        numerical simulation times.
    ResultPath : str
        path to save the simulation result.
    eta : float
        intensity of conformity.

    Returns
    -------
    None.

    '''
    
    #set the parameter
    mu = 1  #set the recoverry probability 
    betas = np.arange(0,1.01,0.01) #set the adoption probability
    
    #construct the array of parameter to run simulation with multithreading  
    args = []    
    for it_num in np.arange(simutimes):
        outbreak_origin = random.choice(list(G.nodes()))    
        args.append([it_num,G,betas,mu,eta,outbreak_origin]) 

    #run the mc with paralell computing 
    pool = Pool(processes=30)    
    results_mc = pool.map(RunOneSimulation,args)
    pk.dump(results_mc, open(ResultPath + '/'+networkName + '_results_mc_'+str(eta)+'.pkl', "wb"))
    ParseResult(results_mc,ResultPath,networkName,eta)
        
if __name__ == '__main__':
    
    #set the path
    NetworkPath = root_path + '/InnovationDiffusion/figure1/network'  #set the network path if run 
    ResultPath = root_path + '/InnovationDiffusion/figure1/result'    #set the result path if run 
    
    #set the parameter
    networkName = ['ca-CondMat.txt', 'email-enron-large.txt','soc-advogato.txt']
    simutimes = 1000    #simulation times 
    Tmax=200    #step length of DMP
    Ndmp = 10   #simulation times of DMP
    alphas = np.arange(0.2,1.01,0.2) #set the alphas 
    
    #load the networks
    for network in networkName:
        print('netname:',network)
        edgelist = np.loadtxt(NetworkPath+'/'+network)
        G = cd.load_network(edgelist)
        #G_updated = NetNameMap(G)
        
        #begin to spread
        for alpha in alphas:
            print('alphas:',alpha)
            NetSpreadDMP(G,network,simutimes,Ndmp,ResultPath,Tmax,alpha)
            NetSpreadMC(G,network,simutimes,ResultPath,alpha)
    
