# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:09:09 2022

@author: Leyang Xue
"""

root_path = 'F:/work/work4_dynamic' #change the current path if run this code

import sys
sys.path.append(root_path+'/InnovationDiffusion')
from utils import coupling_diffusion as cd
import numpy as np
import pandas as pd
import random 
from multiprocessing import Pool
import pickle as pk
from collections import defaultdict

def _simple_transmission_(p):
    '''
    This function can determine whether to adoption 

    Parameters
    ----------
    p : float
        infection rate.

    Returns
    -------
    bool
        wether to infect or not.

    '''
    if random.random() < p:
        return True
    else:
        return False
    
def SIR_G_df(G, p,mu,eta,infecteds, tmin = 0, tmax = float('Inf')):
    '''
    numerical simulation for SAR model on the networks

    Parameters
    ----------
    G : graph
        network.
    p : float
        infection rate.
    mu : float
        recover rate.
    eta : float
        alpha, itensity of comformity effect.
    infecteds : int
        initial seed .
    tmin : int, optional
        start time. The default is 0.
    tmax : TYPE, optional
        maximum infection times. The default is float('Inf').

    Returns
    -------
    S : list
        the number of susceptible nodes over t.
    I : list
        the number of infected nodes over t.
    R : list
        the number of recovered node over t.
    t : list
        time
    '''
    N = G.order()
    if G.has_node(infecteds):
        infecteds = set([infecteds])
        
    t = [0]
    S = [N-len(infecteds)]
    I = [len(infecteds)]
    R = [0]
    
    node_state = defaultdict(lambda : ([tmin], ['S'])) #record the state of nodes 
    susceptible = defaultdict(lambda : True)  #record the susceptible nodes  
    
    for u in infecteds:
        node_state[u][0].append(t[-1])
        node_state[u][1].append('I')
        susceptible[u] = False
    
    while infecteds and t[-1] < tmax-1:
        
        #update the time
        t.append(t[-1]+1) 
        new_infected = set()
        new_recovered = set()
        
        for u in infecteds:
            for v in G.neighbors(u):
                if susceptible[v]:
                   if node_state[u][1][-1] == 'I' and _simple_transmission_(p):
                       node_state[v][1].append('I')
                       node_state[v][0].append(t[-1])
                       susceptible[v] = False
                       new_infected.add(v)
                       
            if  _simple_transmission_(mu):
                node_state[u][1].append('R') 
                node_state[u][0].append(t[-1])
                new_recovered.add(u)
        
        S.append(S[-1] - len(new_infected))
        I.append(len(infecteds)-len(new_recovered)+len(new_infected))
        R.append(R[-1] + len(new_recovered))
        
        for re in new_recovered:
            infecteds.remove(re)
        for ad in new_infected:
            infecteds.add(ad)
        
        if p < 1:
            p += eta * len(new_infected)/N
        else:
            p = 1
        
    return S,I,R,t


def RunSimuBetaC(args):
    '''
    numerical simulation with parallel computation

    Parameters
    ----------
    args : list
        sets of parameters.

    Returns
    -------
    beta_simulation : dict
        simulation results including the number of different state, e.g. S, I, R.

    '''
    it_num,G,betacs,mu,alpha,outbreak_origin = args
    print('simulation:%d'%it_num)

    Nnodes = G.order()
    beta_simulation = {}
        
    for beta in betacs:    
        [S,I,R,t] = SIR_G_df(G,beta,mu,alpha,outbreak_origin)
        s_array= np.array(S)/Nnodes
        i_array = np.array(I)/Nnodes
        r_array = np.array(R)/Nnodes
        t_array = np.array(t)
        beta_simulation[beta]=[s_array,i_array,r_array,t_array]
        
    return beta_simulation

def ParseResult(results,betas,p,ResultPath):
    '''
    Parse the result obtained from the paralell computing
    
    Parameters
    ----------
    results : dicts
        spreading result
    betas : list
        adoption probability 
    p : float
        threshold to determine the outbreak size
    ResultPath : str
        the path to save the result.

    Returns
    -------
    None
    '''
    mint_betadict = {beta:[] for beta in betas}
    rhos_betadict = {beta:[] for beta in betas}
    for each in results:
        for beta in betas:
            [s,i,r,t] = each[beta]
            if r[-1] > p:
                min_time = min(np.where(r > p)[0])#
                mint_betadict[beta].append(min_time) 
            rhos_betadict[beta].append(r[-1])

    pk.dump(rhos_betadict, open(ResultPath+'_results_mc.pkl', "wb"))
    pk.dump(mint_betadict, open(ResultPath+'_results_t.pkl', "wb"))
    
def RhoBetacSpread(G,betacs,alphas,p,file_path):
    '''
    numerical simulation on a network with the given beta and alphac

    Parameters
    ----------
    G : graph
        network.
    betacs : list
        adoption probability.
    alphas : list
        intensity of conformity.
    p : float
        threshold to determined the size of outbreak.
    file_path : str
        path to save the results.

    Returns
    -------
    None.

    '''
    simulation = 10000
    mu = 1
    
    for alpha in alphas:
        #for a given alpha
        args = []
        for i in np.arange(simulation): 
            infected = random.choice(list(G.nodes()))
            args.append([i,G,betacs,mu,alpha,infected])
    
        pool = Pool(processes=30)    
        results = pool.map(RunSimuBetaC,args)
        pool.close()
        pk.dump(results, open(file_path+'.pkl', "wb"))
        ParseResult(results,betacs,p,file_path)

if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #change the current path if run this code
    
    #set the current path 
    network_path = root_path + '/InnovationDiffusion/sfigure3/network'
    ResultPath = root_path + '/InnovationDiffusion/sfigure3/result'
    
    #load the datasets
    statis_data = pd.read_csv(ResultPath + '/complete_datasets_updated.csv')
    p = 0.1
    
    for inx in statis_data.index:
        
        #load the basic information
        network_name = statis_data.iloc[inx]['name']
        dmp_betac = statis_data.iloc[inx]['DMP_betac']
        alphac = statis_data.iloc[inx]['R_alphaC']
        dmp_betacs = [dmp_betac,dmp_betac+0.02,dmp_betac+0.05, dmp_betac+0.1] #for different adoption probability(i.e beta, delayed betac)

        #load the network G
        datasets = np.loadtxt(network_path+'/'+network_name+'.txt', delimiter =',')
        G = cd.load_network(datasets)
        
        #create the path
        file_path = ResultPath + '/' + network_name

        #begin to spread
        RhoBetacSpread(G,dmp_betacs,[alphac],p,file_path)
        