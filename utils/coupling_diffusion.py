# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:24:03 2021

@author: Leyang Xue
"""

import os
import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool 
import pandas as pd 
from collections import defaultdict
import random 
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pickle as pk
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

def load_network(dataset):
    '''
    This function can create the graph from the edgelist 
    Parameters
    ----------
    dataset : list
       network edgelist.

    Returns
    -------
    G : graph
        simple network.
    '''
        
    G = nx.from_edgelist(dataset)
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_G = G.subgraph(largest_cc)
    G=nx.Graph(largest_cc_G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

def load(filename):
   '''
    This function can load some data saved as with a specific format by dump

    Parameters
    ----------
    filename : str 
       path that need to load the file, and the file is saved in dump function.

    Returns
    -------
    TYPE file type, is the same as the previous type 
       file.

    ''' 
   with open(filename, 'rb') as input_file:  
        try:
            return pk.load(input_file)
        
        except EOFError:
            return None
        
def mkdir(path):
    '''
    Create a file folder

    Parameters
    ----------
    path : str
        path.

    Returns
    -------
    None.

    '''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('--create new folder--:'+str(path))
    else:
        print('There is this folder!')

def CriInfectRate(G):
    '''
    calculate the critical infection rate on a uncorrelated network 

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    pc : float
        critical infection rate.

    '''
    sequence = np.array([d for n, d in G.degree()])
    k = np.average(sequence)
    k_2 = np.average(np.power(sequence,2))
    pc = k/(k_2-k)
    
    return pc

def IdentifyCriInfectRate(SimuData):
    '''
    This function can identify the numerical critical point according to the $\chi$ 
    based on the result of numerical simulation 
    
    Parameters
    ----------
    SimuDate : array
       numerical simulation data.

    Returns
    -------
    pc : float
        critical infection rate.
    '''

    nnonzero = np.count_nonzero(SimuData,axis=1)
    cut_point =min(np.argwhere(nnonzero>1))# determine the effective fluaction
    ssum = np.sum(SimuData,axis=1)[cut_point[0]:] 
    savg = ssum/nnonzero[cut_point[0]:]
    
    #std=np.zeros(savg.shape)
    # for inx,i in enumerate(np.arange(cut_point[0],101,1)):
    #     yindex = np.nonzero(SimuData[i,:])[0]
    #     std[inx]= np.sqrt(np.sum(np.power(SimuData[i,yindex]-savg[inx],2))/(nnonzero[i]-1))
    sstd = np.std(SimuData,axis=1)[cut_point[0]:]
    pc_i = np.argmax(sstd/savg)

    betaindex = np.arange(0,101,1)[cut_point[0]:]
    pc=betaindex[pc_i]#/100
     
    return pc

def IdentifySimulationBetac(SimuData):
    '''
    This function can identify the numerical critical point according to the $\chi$ 
    based on the result of numerical simulation 
    
    Parameters
    ----------
    SimuDate : array
       numerical simulation data .

    Returns
    -------
    pc_i : float
        critical infection rate.
    '''

    # nnonzero = np.count_nonzero(SimuData,axis=1)
    # cut_point =min(np.argwhere(nnonzero>1))# determine the effective fluaction
    # ssum = np.sum(SimuData,axis=1)[cut_point[0]:] 
    # savg = ssum/nnonzero[cut_point[0]:]
    # sstd = np.std(SimuData,axis=1)[cut_point[0]:]
    # pc_i = np.argmax(sstd/savg)

    # betaindex = np.arange(0,101,1)[cut_point[0]:]
    # pc=betaindex[pc_i]
    
    savg = np.average(SimuData,axis=0)
    sstd = np.std(SimuData,axis=0)
    var = sstd/savg
    pc_i  = np.argmax(var)
    
    return pc_i

def NetStats(netpath, savepath):
    '''
    The function calculate the basic properities of networks for a group of networks
    
    Parameters
    ----------
    netpath : str
        the path to load the network.
    savepath : str
        the path to save the properities of network.
    
    note that network path need to contain network with .edgelist format
    
    Returns
    -------
    netStats : dataframe
        basic network characteristic statistic.

    '''
    fileName = os.listdir(netpath)
    netStats = pd.DataFrame(index=fileName,columns=['number of node', 'number of edge', 'average degree','average of clustering','average shortest path','assortative','critical infection rate'])

    #load the network
    for name in fileName:
        
        print('process datasets:',name)
        G = nx.read_edgelist(netpath +'/'+name)
        
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_G = G.subgraph(largest_cc)
        G=nx.Graph(largest_cc_G)
        G.remove_edges_from(nx.selfloop_edges(G))
    
        netStats.loc[name,'number of node'] = len(G.nodes())
        netStats.loc[name,'number of edge'] = len(G.edges())
        netStats.loc[name,'average degree'] = 2*netStats.loc[name,'number of edge']/netStats.loc[name,'number of node']
        netStats.loc[name,'average of clustering'] = nx.average_clustering(G)
        netStats.loc[name,'average shortest path'] = nx.average_shortest_path_length(G)
        netStats.loc[name,'assortative'] = nx.degree_assortativity_coefficient(G)
        netStats.loc[name,'critical infection rate'] = CriInfectRate(G)

    netStats.to_csv(savepath+'/NetStatis.csv')
    
    return netStats

def GStats(G,savepath,filename):
    '''
    The function calculate the basic properities of networks for a single network

    Parameters
    ----------
    G : graph
        network.
    savepath : str
        path to save the network statistics.
    filename : str
        the file name of result.

    Returns
    -------
    Stats : Series
        basis statistic of network .
    '''
    Stats ={}
    
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_G = G.subgraph(largest_cc)
    G=nx.Graph(largest_cc_G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    Stats['Nnode'] = len(G.nodes())
    Stats['Nedge'] = len(G.edges())
    Stats['average degree'] = 2*len(G.edges())/len(G.nodes())
    Stats['average of clustering'] = nx.average_clustering(G)
    Stats['average shortest path'] = nx.average_shortest_path_length(G)
    Stats['assortative'] = nx.degree_assortativity_coefficient(G)
    Stats['critical infection rate'] = CriInfectRate(G)
    
    Stats = pd.Series(Stats)
    Stats.to_csv(savepath+'/'+str(filename)+'_Stat.csv')
    
    return Stats

def power_expotent(sequence):
    '''
    estimate the degree exponent using the least square method

    Parameters
    ----------
    sequence : array
        degree sequence.

    Returns
    -------
    gamma : float
        degree expoent.
    alpha : float
        coefficient.

    '''
    k = np.nonzero(sequence)[0]
    pk = sequence[k]
    y = np.log10(pk)
    x = np.log10(k)
    lr = LinearRegression()
    lr.fit(x.reshape(-1,1),y)
    gamma= lr.coef_[0]
    alpha = lr.intercept_
   
    return gamma, alpha

def _simple_transmission_(p):
    '''
    determine to whether to infection or not 

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
    
def SIR_G(G, alpha, mu, infecteds, tmin = 0, tmax = float('Inf')):
    '''
    SIR numerical simulation on a network

    Parameters
    ----------
    G : graph 
        network.
    alpha : float
        infection rate.
    mu : float
        recover probability.
    infecteds : int
        initial infected seed.
    tmin : int, optional
        initial times. The default is 0.
    tmax : int, optional
        maximum times. The default is float('Inf').

    Returns
    -------
    S : list
        the number of nodes in S state over time.
    I : list 
        the number of nodes in I state over time.
    R : TYPE
        the number of nodes in R state over time.

    '''
    N = G.order()
    if G.has_node(infecteds):
        infecteds = set([infecteds])
        
    # sus_over_time = np.ones((N,tmax))
    # inf_over_time = np.zeros((N,tmax))
    # rec_over_time = np.zeros((N,tmax))
    
    t = [0]
    S = [N-len(infecteds)]
    I = [len(infecteds)]
    R = [0]
    
    node_state = defaultdict(lambda : ([tmin], ['S'])) #record the state of nodes 
    susceptible = defaultdict(lambda : True)  #record the susceptible nodes  
    
    # current_susceptible = np.ones(N)
    # current_infected = np.zeros(N) 
    # current_recovered = np.zeros(N)
    
    for u in infecteds:
        node_state[u][0].append(t[-1])
        node_state[u][1].append('I')
        susceptible[u] = False
        # current_susceptible[u] = 0
        # current_infected[u] = 1
        
    # sus_over_time[:,0] = current_susceptible
    # inf_over_time[:,0] = current_infected
        
    while infecteds and t[-1] < tmax-1:#infecteds and 
        
        #update the time
        t.append(t[-1]+1) 
        new_infected = set()
        new_recovered = set()
        
        for u in infecteds:
            for v in G.neighbors(u):
                if susceptible[v]:
                   if node_state[u][1][-1] == 'I' and _simple_transmission_(alpha):
                       node_state[v][1].append('I')
                       node_state[v][0].append(t[-1])
                       susceptible[v] = False
                       new_infected.add(v)
                       
                       # current_susceptible[int(v)] = 0
                       # current_infected[int(v)] = 1
                       
            if  _simple_transmission_(mu):
                node_state[u][1].append('R') 
                node_state[u][0].append(t[-1])
                new_recovered.add(u)
                
                # current_infected[int(u)] = 0
                # current_recovered[int(u)] = 1
        
        # sus_over_time[:,t[-1]] = current_susceptible
        # inf_over_time[:,t[-1]] = current_infected
        # rec_over_time[:,t[-1]] = current_recovered
        S.append(S[-1] - len(new_infected))
        I.append(len(infecteds)-len(new_recovered)+len(new_infected))
        R.append(R[-1] + len(new_recovered))
        
        for re in new_recovered:
            infecteds.remove(re)
        for ad in new_infected:
            infecteds.add(ad)
    
    return S,I,R

def runSIRdf(args):
    
    #this function is used to called by other funtion with paralell computation
    
    it,G,p,mu,eta,infected = args
    print('run %d times'%it)
    [S,I,R]= SIR_G_df(G,p,mu,eta,infected)
    
    return S,I,R

def SIR_G_df(G, p,mu,eta,infecteds, tmin = 0, tmax = float('Inf')):
    '''
    numerical simulation of SAR model on networks

    Parameters
    ----------
    G : graph
        network.
    p : float
        infection rate.
    mu : float
        recover rate.
    eta : float
        feedback parameter used to determine the strength of conformity.
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
        
    return S,I,R

def SIR_G_df_Nodes(G, p,mu,eta,infecteds, tmin = 0, tmax = float('Inf')):
    """    
    numerical simulation of SAR model on networks, but different from SIR_G_df
    This function return the nodes staying different state instead of the number of nodes
    
    Parameters
    ----------
    G : graph
        network.
    p : float
        infection rate.
    mu : float
        recover rate.
    eta : float
        feedback parameter used to determine the strength.
    infecteds : int
        initial seed .
    tmin : int, optional
        start time. The default is 0.
    tmax : TYPE, optional
        maximum infection times. The default is float('Inf').

    Returns
    -------
    S : list
        susceptible nodes over t.
    I : list
        infected nodes over t.
    R : list
        recovered node over t.
    """
    N = G.order()
    if G.has_node(infecteds):
        infecteds = set([infecteds])

    t = [0]
    S = [list(set(G.nodes())-set(infecteds))]
    I = [list(infecteds)]
    R = []
    
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
                
        S.append(list(set(S[-1])-set(new_infected)))
        I.append(list((set(infecteds)-set(new_recovered)) | set(new_infected)))
        R.append(list(set(new_recovered)))
        
        for re in new_recovered:
            infecteds.remove(re)
        for ad in new_infected:
            infecteds.add(ad)
        
        if p < 1:
            p += eta * len(new_infected)/N
        else:
            p = 1   
        
    return S,I,R

def DMP(alpha,mu,edgelist,outbreak_origin,Tmax,directed):
    '''
    numerical reuslts of SIR model for DMP method 
    
    Parameters
    ----------
    alpha : float
        infection rate.
    mu : float
        recover rate.
    edgelist : list
        network edgelist.
    outbreak_origin : int
        initial outbreak original.
    Tmax : int
        maximum infection time.
    directed : Bool
         True if the network is directed, otherwise False.

    Returns
    -------
    sus_over_time : array
        the probability of nodes staying in susceptible state over time 
    inf_over_time : array
        the probability of nodes staying in infected state over time 
    rec_over_time : array 
        the probability of nodes staying in recovered state over time 
    '''
    edgelist = np.array(edgelist, dtype=int)
    alpha = float(alpha)
    mu = float(mu)
    directed = bool(directed)
    
    #1. check the variable
    nodes = set(edgelist[:,0]) | set(edgelist[:,1])
    Nnodes = len(nodes)
    assert nodes == set(range(Nnodes)), "nodes must be named continuously from 0 to 'number of nodes'-1 "
    assert alpha >= 0 and alpha <= 1, "infection probability incorrect"
    assert mu >= 0 and mu <= 1, "recovery probability incorrect"
    if isinstance(outbreak_origin, int):
         assert outbreak_origin >= 0 and outbreak_origin <  Nnodes, "0 <= outbreak origin < 'number of nodes'"


    #2. transform the undirected edgelist to directed edgelist if the directed == false
    if not directed:
        edgelist_directed = np.zeros((len(edgelist)*2, 2))
        for idx in range(len(edgelist)):
            u, v = edgelist[idx]
            edgelist_directed[2*idx, :] = u, v
            edgelist_directed[2*idx + 1, :] = v, u
        edgelist = edgelist_directed.astype(int)
        
    #3. create the directed network and obtain the number of nodes and edges, idx of edges 
    G = nx.DiGraph()
    G.add_edges_from([(u,v) for u,v in edgelist])
    assert nx.number_of_selfloops(G) == 0, "self loops are not allowed"
    out_neighbours = G.succ #out neighbors
    in_neighbours = G.pred #in neighbors
    Nedges = G.number_of_edges()
    Nnodes = G.number_of_nodes()
    edge_to_idx = {edge: idx for idx, edge in enumerate(G.edges())}
    reciprical_link_exists = {edge_to_idx[ u, v ]: True if G.has_edge(v,u) else False for u,v in G.edges()} #for cavity
    
    #4.create the variable array
    sus_over_time = np.zeros((Nnodes, Tmax)) #probability: node is susceptible at time 0 <= t < Tmax
    inf_over_time = np.zeros((Nnodes, Tmax)) #probability: node is infected at time 0 <= t < Tmax
    rec_over_time = np.zeros((Nnodes, Tmax)) #probability: node is recovered at time 0 <= t < Tmax

    sus     = np.ones(Nedges) # conditional probability: node is susceptible given neighbor is in cavity state
    sus_new = np.ones(Nedges) # temporal buffer
    theta   = np.ones(Nedges) # conditional probability: no disease has been transmitted across edge given neighbor is in cavity state
    phi     = np.zeros(Nedges) # conditional probability: node is infected and has not trasmitted infection to neighbor given neighbor is in cavity state
    
    #5. t0,initial settting, mark edge that leave initially infected node
    init_idx = np.array([edge_to_idx[node, cavity] for (node, cavity) in G.edges() if node == outbreak_origin], dtype=int)
    # initial condition
    sus[init_idx] = 0.
    sus_new[init_idx] = 0.
    phi[init_idx] = 1.
    
    susceptible = np.ones(Nnodes)  # susceptible = sus_over_time[ time ]
    susceptible[outbreak_origin] = 0.
    infected = np.zeros(Nnodes) #infected = inf_over_time[ time ]
    infected[outbreak_origin] = 1.
    recovered = np.zeros(Nnodes) # recovered = rec_over_time[ time ]
    
    sus_over_time[:,0] = susceptible
    inf_over_time[:,0] = infected
  
    for time in range(Tmax-1):
        #update at a time instance
        for target in list(G.nodes):
            # outbreak origin cannot be infected
            if target != outbreak_origin:
                susceptible[target] = 1.
                # go through all (static) predecessors of target
                for source in in_neighbours[target]:
                    edge_idx = edge_to_idx[source, target]
                    # infection transmission (update theta for active edges only)
                    # do not account for non-backtracking property here
                    # update theta
                    theta[edge_idx] -= alpha * phi[edge_idx]
                    susceptible[target] *= theta[ edge_idx ]

                # account for non-backtracking property
                for cavity in out_neighbours[target]:
                    edge_idx = edge_to_idx[target, cavity]
                    sus_new[edge_idx] = susceptible[ target ]
                    # discount transmission probability along backtracking contact: cavity -> target
                    if reciprical_link_exists[edge_idx]:
                        reciprical_edge_idx = edge_to_idx[cavity, target]
                        sus_new[edge_idx] /= theta[ reciprical_edge_idx] #calculate the prbability that node is in susceptible state in cavity 
                            

        # update edge-based quantities only for contact-based model
        phi *= (1. - mu) * (1. - alpha)
        phi +=  sus - sus_new
        sus = sus_new.copy()
        
        recovered += infected * mu
        infected = 1. - recovered - susceptible
        
        sus_over_time[:, time + 1] = susceptible
        inf_over_time[:, time + 1] = infected
        rec_over_time[:, time + 1] = recovered
    
    return sus_over_time, inf_over_time, rec_over_time

def DMP_feedback(alpha,mu,eta,edgelist,outbreak_origin,Tmax,directed):
    '''
    numerical reuslts of SAR model for DMP method 

    Parameters
    ----------
    alpha : float
        infection rate.
    mu : float
        recover rate.
    eta : float
        feedback parameters, it is used to determine the feedback strength.
    edgelist : list
        network edgelist.
    outbreak_origin : int
        initial outbreak original.
    Tmax : int
        maximum infection time.
    directed : Bool
         True if the network is directed, otherwise False.

    Returns
    -------
    sus_over_time : array
        the probability of nodes staying in susceptible state over time 
    inf_over_time : array
        the probability of nodes staying in infected state over time 
    rec_over_time : array 
        the probability of nodes staying in recovered state over time 
  
    '''
    edgelist = np.array(edgelist, dtype=int)
    alpha = float(alpha)
    mu = float(mu)
    directed = bool(directed)
    
    #1. check the variable
    nodes = set(edgelist[:,0]) | set(edgelist[:,1])
    Nnodes = len(nodes)
    assert nodes == set(range(Nnodes)), "nodes must be named continuously from 0 to 'number of nodes'-1 "
    assert alpha >= 0 and alpha <= 1, "infection probability incorrect"
    assert mu >= 0 and mu <= 1, "recovery probability incorrect"
    if isinstance(outbreak_origin, int):
         assert outbreak_origin >= 0 and outbreak_origin <  Nnodes, "0 <= outbreak origin < 'number of nodes'"


    #2. transform the undirected edgelist to directed edgelist if the directed == false
    if not directed:
        edgelist_directed = np.zeros((len(edgelist)*2, 2))
        for idx in range(len(edgelist)):
            u, v = edgelist[idx]
            edgelist_directed[2*idx, :] = u, v
            edgelist_directed[2*idx + 1, :] = v, u
        edgelist = edgelist_directed.astype(int)
        
    #3. create the directed network and obtain the number of nodes and edges, idx of edges 
    G = nx.DiGraph()
    G.add_edges_from([(u,v) for u,v in edgelist])
    assert nx.number_of_selfloops(G) == 0, "self loops are not allowed"
    out_neighbours = G.succ #out neighbors
    in_neighbours = G.pred #in neighbors
    Nedges = G.number_of_edges()
    Nnodes = G.number_of_nodes()
    edge_to_idx = {edge: idx for idx, edge in enumerate(G.edges())}
    reciprical_link_exists = {edge_to_idx[ u, v ]: True if G.has_edge(v,u) else False for u,v in G.edges()} #for cavity
    
    #4.create the variable array
    sus_over_time = np.zeros((Nnodes, Tmax)) #probability: node is susceptible at time 0 <= t < Tmax
    inf_over_time = np.zeros((Nnodes, Tmax)) #probability: node is infected at time 0 <= t < Tmax
    rec_over_time = np.zeros((Nnodes, Tmax)) #probability: node is recovered at time 0 <= t < Tmax

    sus     = np.ones(Nedges) # conditional probability: node is susceptible given neighbor is in cavity state
    sus_new = np.ones(Nedges) # temporal buffer
    theta   = np.ones(Nedges) # conditional probability: no disease has been transmitted across edge given neighbor is in cavity state
    phi     = np.zeros(Nedges) # conditional probability: node is infected and has not trasmitted infection to neighbor given neighbor is in cavity state
    
    #5. t0,initial settting, mark edge that leave initially infected node
    init_idx = np.array([edge_to_idx[node, cavity] for (node, cavity) in G.edges() if node == outbreak_origin], dtype=int)
    # initial condition
    sus[init_idx] = 0.
    sus_new[init_idx] = 0.
    phi[init_idx] = 1.
    
    susceptible = np.ones(Nnodes)  # susceptible = sus_over_time[ time ]
    susceptible[outbreak_origin] = 0.
    infected = np.zeros(Nnodes) #infected = inf_over_time[ time ]
    infected[outbreak_origin] = 1.
    recovered = np.zeros(Nnodes) # recovered = rec_over_time[ time ]
    
    sus_over_time[:,0] = susceptible
    inf_over_time[:,0] = infected
    
    for time in range(Tmax-1):
        #update at a time instance
        if alpha < 1:
            alpha += eta*np.sum(infected)/Nnodes 
        else:
            alpha =  1
            
        for target in list(G.nodes):
            # outbreak origin cannot be infected
            if target != outbreak_origin:
                susceptible[target] = 1.
                # go through all (static) predecessors of target
                cavitydict ={}
                for source in in_neighbours[target]:
                    edge_idx = edge_to_idx[source, target]
                    # infection transmission (update theta for active edges only)
                    # do not account for non-backtracking property here
                    # update theta
                    theta[edge_idx] -= alpha * phi[edge_idx] #theta^{source->target} quantatities
                    if theta[edge_idx] < 0:
                       theta[edge_idx] = 0
                    cavitydict[edge_idx] = theta[edge_idx] #theta^{source->target}         
                    susceptible[target] *= theta[edge_idx] #P_target

                # account for non-backtracking property
                for cavity in out_neighbours[target]:
                    edge_idx = edge_to_idx[target, cavity]
                    sus_new[edge_idx] = susceptible[target]
                    # discount transmission probability along backtracking contact: cavity -> target
                    if reciprical_link_exists[edge_idx]:
                        reciprical_edge_idx = edge_to_idx[cavity, target]
                        if theta[reciprical_edge_idx] != 0:
                            sus_new[edge_idx] /= theta[reciprical_edge_idx] #calculate the prbability that node is in susceptible state in cavity 
                        else:
                            for edgeidx in cavitydict.keys():
                                if edgeidx != reciprical_edge_idx:
                                    sus_new[edge_idx] *= cavitydict[edgeidx]
                                    
        # update edge-based quantities only for contact-based model
        phi *= (1. - mu) * (1. - alpha)
        phi +=  sus - sus_new
        phi[phi>1.0] = 1.0
            
        sus = sus_new.copy()
        
        recovered += infected * mu
        infected = 1. - recovered - susceptible
        
        sus_over_time[:, time + 1] = susceptible
        inf_over_time[:, time + 1] = infected
        rec_over_time[:, time + 1] = recovered
    
    return sus_over_time, inf_over_time, rec_over_time

def DMP_Betat(alpha,mu,eta,edgelist,outbreak_origin,Tmax,directed):
    '''
    numerical reuslts of SAR model for DMP method, but it return the time serise of beta  

    Parameters
    ----------
    alpha : float
        infection rate.
    mu : float
        recover rate.
    eta : float
        feedback parameters, it is used to determine the feedback strength.
    edgelist : list
        network edgelist.
    outbreak_origin : int
        initial outbreak original.
    Tmax : int
        maximum infection time.
    directed : Bool
         True if the network is directed, otherwise False.

    Returns
    -------
    sus_over_time : array
        the probability of nodes staying in susceptible state over time 
    inf_over_time : array
        the probability of nodes staying in infected state over time 
    rec_over_time : array 
        the probability of nodes staying in recovered state over time 
  
    '''
    edgelist = np.array(edgelist, dtype=int)
    alpha = float(alpha)
    mu = float(mu)
    directed = bool(directed)
    
    #1. check the variable
    nodes = set(edgelist[:,0]) | set(edgelist[:,1])
    Nnodes = len(nodes)
    assert nodes == set(range(Nnodes)), "nodes must be named continuously from 0 to 'number of nodes'-1 "
    assert alpha >= 0 and alpha <= 1, "infection probability incorrect"
    assert mu >= 0 and mu <= 1, "recovery probability incorrect"
    if isinstance(outbreak_origin, int):
         assert outbreak_origin >= 0 and outbreak_origin <  Nnodes, "0 <= outbreak origin < 'number of nodes'"


    #2. transform the undirected edgelist to directed edgelist if the directed == false
    if not directed:
        edgelist_directed = np.zeros((len(edgelist)*2, 2))
        for idx in range(len(edgelist)):
            u, v = edgelist[idx]
            edgelist_directed[2*idx, :] = u, v
            edgelist_directed[2*idx + 1, :] = v, u
        edgelist = edgelist_directed.astype(int)
        
    #3. create the directed network and obtain the number of nodes and edges, idx of edges 
    G = nx.DiGraph()
    G.add_edges_from([(u,v) for u,v in edgelist])
    assert nx.number_of_selfloops(G) == 0, "self loops are not allowed"
    out_neighbours = G.succ #out neighbors
    in_neighbours = G.pred #in neighbors
    Nedges = G.number_of_edges()
    Nnodes = G.number_of_nodes()
    edge_to_idx = {edge: idx for idx, edge in enumerate(G.edges())}
    reciprical_link_exists = {edge_to_idx[ u, v ]: True if G.has_edge(v,u) else False for u,v in G.edges()} #for cavity
    
    #4.create the variable array
    sus_over_time = np.zeros((Nnodes, Tmax)) #probability: node is susceptible at time 0 <= t < Tmax
    inf_over_time = np.zeros((Nnodes, Tmax)) #probability: node is infected at time 0 <= t < Tmax
    rec_over_time = np.zeros((Nnodes, Tmax)) #probability: node is recovered at time 0 <= t < Tmax

    sus     = np.ones(Nedges) # conditional probability: node is susceptible given neighbor is in cavity state
    sus_new = np.ones(Nedges) # temporal buffer
    theta   = np.ones(Nedges) # conditional probability: no disease has been transmitted across edge given neighbor is in cavity state
    phi     = np.zeros(Nedges) # conditional probability: node is infected and has not trasmitted infection to neighbor given neighbor is in cavity state
    
    #5. t0,initial settting, mark edge that leave initially infected node
    init_idx = np.array([edge_to_idx[node, cavity] for (node, cavity) in G.edges() if node == outbreak_origin], dtype=int)
    # initial condition
    sus[init_idx] = 0.
    sus_new[init_idx] = 0.
    phi[init_idx] = 1.
    
    susceptible = np.ones(Nnodes)  # susceptible = sus_over_time[ time ]
    susceptible[outbreak_origin] = 0.
    infected = np.zeros(Nnodes) #infected = inf_over_time[ time ]
    infected[outbreak_origin] = 1.
    recovered = np.zeros(Nnodes) # recovered = rec_over_time[ time ]
    
    sus_over_time[:,0] = susceptible
    inf_over_time[:,0] = infected
    
    betat = {}
    betat[0] = alpha 

    for time in range(Tmax-1):
        #update at a time instance
        if alpha < 1:
            alpha += eta*np.sum(infected)/Nnodes 
        else:
            alpha =  1
            
        betat[time+1] = alpha 
        
        for target in list(G.nodes):
            # outbreak origin cannot be infected
            if target != outbreak_origin:
                susceptible[target] = 1.
                # go through all (static) predecessors of target
                cavitydict ={}
                for source in in_neighbours[target]:
                    edge_idx = edge_to_idx[source, target]
                    # infection transmission (update theta for active edges only)
                    # do not account for non-backtracking property here
                    # update theta
                    theta[edge_idx] -= alpha * phi[edge_idx]
                    cavitydict[edge_idx] = theta[edge_idx]         
                    susceptible[target] *= theta[edge_idx]

                # account for non-backtracking property
                for cavity in out_neighbours[target]:
                    edge_idx = edge_to_idx[target, cavity]
                    sus_new[edge_idx] = susceptible[target]
                    # discount transmission probability along backtracking contact: cavity -> target
                    if reciprical_link_exists[edge_idx]:
                        reciprical_edge_idx = edge_to_idx[cavity, target]
                        if theta[ reciprical_edge_idx] != 0:
                            sus_new[edge_idx] /= theta[reciprical_edge_idx] #calculate the prbability that node is in susceptible state in cavity 
                        else:
                            for edgeidx in cavitydict.keys():
                                if edgeidx != reciprical_edge_idx:
                                    sus_new[edge_idx] *= cavitydict[edgeidx]
                                    
        # update edge-based quantities only for contact-based model
        phi *= (1. - mu) * (1. - alpha)
        phi +=  sus - sus_new
        sus = sus_new.copy()
        
        recovered += infected * mu
        infected = 1. - recovered - susceptible
        
        sus_over_time[:, time + 1] = susceptible
        inf_over_time[:, time + 1] = infected
        rec_over_time[:, time + 1] = recovered
    
    return sus_over_time, inf_over_time, rec_over_time, betat


def GeneratePowerlawSq(size,gamma,min_degree=2):
    '''
    This function can generate the degree sequence with degree exponent i.e. gamma

    Parameters
    ----------
    size : int
        the size of network.
    gamma : float
        degree exponent.
    min_degree : int, optional
        the minimum degree in the network. The default is 2.

    Returns
    -------
    degree_sequence : list
        degree sequence.

    '''
    degree_sequence = []
    max_degree = int(np.power(size,0.5))
    i = 0
    while i < size:
         k = int(round(np.power(np.random.rand(), 1/(1-gamma)),0))
         if k > min_degree and  k < max_degree:
             degree_sequence.append(k)
             i = i+1
    
    if sum(degree_sequence) % 2 == 0:
        return degree_sequence
    else:
        index = np.random.choice(size)
        degree_sequence[index] = degree_sequence[index]+1
        return degree_sequence
 
def calculate_frequence(number_of_infected):
    '''
    calculate the frequence of sequence

    Parameters
    ----------
    number_of_infected : array
        an array.

    Returns
    -------
    freq : dict 
        the ratio of different element in the sequence.

    '''
    L = list(number_of_infected)
    set_L = set(L)
    freq = {i:L.count(i)/len(L) for i in set_L} 
    
    return freq
       
def generate_LFR_benchmark(n,tau1,tau2,mu,min_degree):
    
    community_G = nx.LFR_benchmark_graph(n,tau1,tau2,mu,min_degree = min_degree)

    return community_G
    
def generate_different_LFR_benchmark(n,taus,mu,min_degree,save_file):
    
    for tau in taus:
        print('community size distribution exponent:', tau)
        print('network power exponent:', tau)
        for m in mu:
            print('community clarity parameter:',m)
            community_benchmark = generate_LFR_benchmark(n, tau, tau, m,min_degree)  
            nx.write_adjlist(community_benchmark, save_file+'/'+str(round(tau,2))+'_'+str(round(tau,2))+'_'+str(round(m,2))+'.adjlist')

def RunOneSimulation(args):
    '''
    This function could be run with paralell computing, return the result of SIR and SAR
    
    Parameters
    ----------
    args : list
        parameter sets.

    Returns
    -------
    r_alphas : list
        infected ratio.
    '''
    
    it_num,G,alphas,mu,eta,outbreak_origin = args
    Nnodes = G.order()
    print('simulation:%d'%it_num)
    
    r_alphas = []
    r_alphas_df = []
    for j, alpha in enumerate(alphas):
        [S,I,R] = SIR_G(G,alpha,mu,outbreak_origin)
        [s,i,r] = SIR_G_df(G,alpha,mu,eta,outbreak_origin)
        r_alphas.append(R[-1]/Nnodes) 
        r_alphas_df.append(r[-1]/Nnodes)
        
    return r_alphas, r_alphas_df

def ParseResult(results,save_path,file):
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
    rhos_list,rhos_df_list = [],[]
    for rhos,rhos_df in results:
        rhos_list.append(rhos)
        rhos_df_list.append(rhos_df)
        
    spreading = np.array(rhos_list) 
    spreading_df = np.array(rhos_df_list)

    np.savetxt(save_path+'/'+file + '_spreading.csv',spreading, delimiter = ',')
    np.savetxt(save_path+'/'+file+ '_spreading_df.csv',spreading_df, delimiter = ',')
    
    return spreading, spreading_df
    
def NetSpread(G,eta,simutimes,ResultPath,networkName):
    '''
    Spreading on a given network 

    Parameters
    ----------
    G : networkx 
        network.
    simutimes : int
        simulation times.
    ResultPath : str
        the path to save.
    networkName : str
        file name.

    Returns
    -------
    spreading : array
       SIR numerical simulation.
    spreading_df : array
       SAR numerical simulation.

    '''
    #set the spreading parameter
    mu = 1
    betas = np.arange(0,1.01,0.01)

    args = []    
    for it_num in np.arange(simutimes):
        outbreak_origin = random.choice(list(G.nodes()))    
        args.append([it_num,G,betas,mu,eta,outbreak_origin]) 

    #1. run the mc with paralell computing 
    pool = Pool(processes=40)    
    results_mc = pool.map(RunOneSimulation,args)
    pk.dump(results_mc, open(ResultPath + '/'+networkName + '_results_mc.pkl', "wb"))
    [spreading, spreading_df] = ParseResult(results_mc,ResultPath,networkName)
    
    return spreading, spreading_df

def Generate2d(size,save_path):
    '''
    Generating the 2d graph

    Parameters
    ----------
    size : int
        network size.
    save_path : str
        the path to save the file.

    Returns
    -------
    None.

    '''
    G = nx.grid_2d_graph(size,size,periodic=True)
    nx.write_edgelist(G, save_path+'/2d.edgelist')
    
def GenerateER(size,ks,save_path):
    '''
    Generating a group of ER network

    Parameters
    ----------
    size : int
        the number of node.
    k : list
        a list consiting of average degree.
    save_path : str
        path to save the network.

    Returns
    -------
    None.

    '''
    p = ks/size
    for i,each in enumerate(p):
        print("average degree:", ks[i])
        
        G = nx.erdos_renyi_graph(size,p)
        lcc= max(nx.connected_components(G),key=len)
        while len(lcc)!=size:
            G = nx.erdos_renyi_graph(size,p)
            lcc= max(nx.connected_components(G),key=len)
            
        nx.write_edgelist(G, save_path+'/ER_'+str(ks[i])+'.edgelist')


def GenerateWS(size,ks,p,save_path):
    '''
    Generating the WS network 

    Parameters
    ----------
    size : int
        the number of nodes.
    ks : list
        a list consiting of average degree, where each node is joined with its k nearest neighbors.
    p : float
        the probability of rewiring each edge.
    save_path : str
        path to save the network.

    Returns
    -------
    None.

    '''
    for each in ks:
        print('average degree:', each)
        G = nx.watts_strogatz_graph(size,each,p)
        nx.write_edgelist(G, save_path+'/WS_'+str(each)+'.edgelist')

def GenerateBA(size,ms,save_path):
    '''
    Generating the BA network 

    Parameters
    ----------
    size : int
       network size .
    ms : list
        a list consiting of the number of edge added to the network .
    save_path : str
       path to save the network.

    Returns
    -------
    None.

    '''
    for each in ms:
        print('average degree:', 2*each)
        G = nx.barabasi_albert_graph(size, each)
        nx.write_edgelist(G, save_path+'/BA_'+str(2*each)+'.edgelist')  
    
    #return G

def GenerateSF(size,gamma,min_degree,path,name):
    '''
    Generating the Scale-free network 

    Parameters
    ----------
    size : int
        the size of network.
    gamma : float
        the powerlaw exponent.
    min_degree : int
        the minimum degree in the network.
    path : str
        the path to save the network.

    Returns
    -------
    G : graph 
        generated network with powerlaw exponent gamma.

    '''
    NetStat= pd.DataFrame(columns=['nodes','edges','average degree','pc'])    
    largest=[]
    while len(largest) != size:
        degree_sq = GeneratePowerlawSq(size,gamma,min_degree)
        G = nx.configuration_model(degree_sq)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        largest = max(nx.connected_components(G), key=len)
        
        if len(largest) == size:
            print('Generating the ScaleFree network')

    NetStat.loc[str(gamma),'nodes'] = len(G.nodes)
    NetStat.loc[str(gamma),'edges'] = len(G.edges)
    NetStat.loc[str(gamma),'average degree'] = np.average(degree_sq)
    NetStat.loc[str(gamma),'pc'] = CriInfectRate(G)
    
    NetworkName = name + '_' + str(size)+'_'+str(min_degree)+'_'+str(gamma)
    NetStat.to_csv(path+'/'+NetworkName+'_inf.csv')
    nx.write_edgelist(G, path+'/'+NetworkName+'.edgelist')
    
    return G

def fractal_model(generation,m,x,e):
    
    """
    Returns the fractal model introduced by 
    Song, Havlin, Makse in Nature Physics 2, 275.
    generation = number of generations
    m = number of offspring per node
    x = number of connections between offsprings
    e = probability that hubs stay connected
    1-e = probability that x offsprings connect.
    If e=1 we are in MODE 1 (pure small-world).
    If e=0 we are in MODE 2 (pure fractal).
    """
    
    G=nx.Graph()
    G.add_edge(0,1) #This is the seed for the network (generation 0)
    node_index = 2
    for n in np.arange(1,generation+1):
        all_links = list(G.edges()) 
        while all_links:
            link = all_links.pop()
            new_nodes_a = range(node_index,node_index + m)
            #random.shuffle(new_nodes_a)
            node_index += m
            new_nodes_b = range(node_index,node_index + m)
            #random.shuffle(new_nodes_b)
            node_index += m
            G.add_edges_from([(link[0],node) for node in new_nodes_a])
            G.add_edges_from([(link[1],node) for node in new_nodes_b])
            repulsive_links = list(zip(new_nodes_a,new_nodes_b))
            G.add_edges_from([repulsive_links.pop() for i in range(x-1)])
            
            if random.random() > e:
                G.remove_edge(link[0],link[1])
                r_link = repulsive_links.pop()
                G.add_edge(r_link[0],r_link[1])
    return G


def JointDegreeDist(G,name):
    '''
    

    Parameters
    ----------
    G : graph
        network.
    name : str
        network name.

    Returns
    -------
    JointDistri: array
        joint degree distribution of network 
    knns: array
        average degree
    ''' 
    degree_dict = {item[0]: item[1] for item in nx.degree(G)}     
    degree = np.array(nx.degree_histogram(G))
    k = np.nonzero(degree)[0]
    # degreeCorr = np.zeros((len(k),len(k)))
    Degree_Corr_matrix= np.zeros((len(k),len(k)))
    
    KtoInx = {each:i for i,each in enumerate(k)}
    for edges in G.edges():
       inx = KtoInx[degree_dict[edges[0]]]
       iny = KtoInx[degree_dict[edges[1]]]
       Degree_Corr_matrix[inx,iny] += 1
       Degree_Corr_matrix[iny,inx] += 1
       
    JointDistri = Degree_Corr_matrix/len(G.edges)
    binary = JointDistri.copy()
    binary[binary!=0]=1
    
    knn = nx.average_neighbor_degree(G)
    knnSeries = pd.Series(knn)
    
    DegreeTONodeInx=defaultdict(lambda:[])
    for each in degree_dict.items():
        DegreeTONodeInx[each[1]].append(each[0])
           
    knns = np.zeros(len(k))
    for each in k:
       Nodeinx = DegreeTONodeInx[each]
       kinx = KtoInx[each]
       knns[kinx] = np.average(knnSeries[Nodeinx])

    return JointDistri,knns   

def Xulvi_Brunet_Sokolov(args):
    '''
    Xulvi_Brunet_Sokolov algorithms 
    
    Parameters
    ----------
    G0 : graph
        original network.
    p : float
       the ratio to swap the edges 
    assort : bool
       turning the network from current to assortative if True,
       dissortative otherwise.
    NetworkName : str
        network name.
    ResultPath : str
        the path to save the network.

    Returns
    -------
    None.
    '''
    
    G0,p,assort,NetworkName,ResultPath = args

    G = G0.copy()
    Maxstep = G.size()
    n = 0
    swap_times = 0
    
    while n < Maxstep: 
        
        degree = G.degree()
        edge_set = list(G.edges()) 
    
        link_1 = random.choice(edge_set)
        link_2 = random.choice(edge_set)
        node_set = set([link_1[0],link_1[1],link_2[0],link_2[1]])
        
        print('P value network:%f,try %d time edges swap %d edges in total %d times'%(p,n,swap_times,Maxstep))
        if len(node_set) == 4: #ensure that the four index is different
        
            group_index = list(node_set)#four index
            group_degree= np.array(list(map(lambda x:degree[x], group_index)))
            order_index = np.argsort(group_degree)#order generated from large to small
            
            if np.random.rand() < p:
                if assort == True: #assortative
                    new_index_1 = (group_index[order_index[0]],group_index[order_index[1]])
                    new_index_2 = (group_index[order_index[2]],group_index[order_index[3]])
                    if G.has_edge(new_index_1[0],new_index_1[1]) == 0 and G.has_edge(new_index_2[0], new_index_2[1]) == 0:
                        G.remove_edge(link_1[0], link_1[1])
                        G.remove_edge(link_2[0], link_2[1])
                        G.add_edge(new_index_1[0],new_index_1[1])
                        G.add_edge(new_index_2[0],new_index_2[1])
                        swap_times = swap_times+1
                    n=n+1

                else:   #disassortative
                    new_index_1 = (group_index[order_index[0]],group_index[order_index[3]])
                    new_index_2 = (group_index[order_index[1]],group_index[order_index[2]])
                    if G.has_edge(new_index_1[0],new_index_1[1]) == 0 and G.has_edge(new_index_2[0],new_index_2[1]) == 0:  
                        G.remove_edge(link_1[0], link_1[1])
                        G.remove_edge(link_2[0], link_2[1])
                        G.add_edge(new_index_1[0],new_index_1[1])
                        G.add_edge(new_index_2[0],new_index_2[1])
                        swap_times = swap_times+1
                    n=n+1
            else:
               sample = np.random.choice(group_index,2,replace=False)
               new_edge_1 = set(sample)
               new_edge_2 = set(group_index)-new_edge_1
               edge1 = list(new_edge_1)
               edge2 = list(new_edge_2)
               if G.has_edge(edge1[0],edge1[1]) == 0 and G.has_edge(edge2[0],edge2[1]) == 0:
                   G.remove_edge(link_1[0], link_1[1])
                   G.remove_edge(link_2[0], link_2[1])
                   G.add_edge(edge1[0],edge1[1])
                   G.add_edge(edge2[0],edge2[1])
               n=n+1
        
    nx.write_edgelist(G, ResultPath+'/'+NetworkName+'.edgelist')

def half_incidence(graph, ordering='blocks', return_ordering=False):
    """Return the 'half incidence' matrices of the graph.

    The resulting matrices have shape of (n, 2m), where n is the number of
    nodes and m is the number of edges.

    Params
    ------

    graph (nx.Graph): The graph.

    ordering (str): If 'blocks' (default), the two columns corresponding to
    the i'th edge are placed at i and i+m. That is, choose an arbitarry
    direction for each edge in the graph. The first m columns correspond to
    this orientation, while the latter m columns correspond to the reversed
    orientation. Columns in both blocks are sorted following graph.edges.
    If 'consecutive', the first two columns correspond to the two
    orientations of the first edge, the third and fourth row are the two
    orientations of the second edge, and so on. In general, the two columns
    for the i'th edge are placed at 2i and 2i+1. If 'custom', parameter
    custom must be a dictionary of pairs of the form (idx, (i, j)) where
    the key idx maps onto a 2-tuple of indices where the edge must fall.

    custom (dict): Used only when ordering is 'custom'.

    return_ordering (bool): If True, return a function that maps an edge id
    to the column placement. That is, if ordering=='blocks', return the
    function lambda x: (x, m+x), if ordering=='consecutive', return the
    function lambda x: (2*x, 2*x + 1). If False, return None.


    Returns
    -------

    (source, target), or (source, target, ord_function) if return_ordering
    is True.


    Notes
    -----

    Assumes the nodes are labeled by consecutive integers starting at 0.

    """
    numnodes = graph.order()
    numedges = graph.size()

    if ordering == 'blocks':
        src_pairs = lambda i, u, v: [(u, i), (v, numedges + i)]
        tgt_pairs = lambda i, u, v: [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        src_pairs = lambda i, u, v: [(u, 2*i), (v, 2*i + 1)]
        tgt_pairs = lambda i, u, v: [(v, 2*i), (u, 2*i + 1)]
    if isinstance(ordering, dict):
        src_pairs = lambda i, u, v: [(u, ordering[i][0]), (v, ordering[i][1])]
        tgt_pairs = lambda i, u, v: [(v, ordering[i][0]), (u, ordering[i][1])]

    def make_coo(make_pairs):
        """Make a sparse 0-1 matrix.

        The returned matrix has a positive entry at each coordinate pair
        returned by make_pairs, for all (idx, node1, node2) edge triples.

        """
        coords = [pair
                  for idx, (node1, node2) in enumerate(graph.edges())
                  for pair in make_pairs(idx, node1, node2)]
        data = np.ones(len(coords))
        return sparse.coo_matrix((data, list(zip(*coords))),
                                 shape=(numnodes, 2*numedges))

    source = make_coo(src_pairs).asformat('csr')
    target = make_coo(tgt_pairs).asformat('csr')

    if return_ordering:
        if ordering == 'blocks':
            ord_func = lambda x: (x, numedges+x)
        elif ordering == 'consecutive':
            ord_func = lambda x: (2*x, 2*x+1)
        elif isinstance(ordering, dict):
            ord_func = lambda x: ordering[x]
        return source, target, ord_func
    else:
        return source, target   
    
def nb_matrix(graph, aux=False, ordering='blocks', return_ordering=False):
    """Return NB-matrix of a graph.

    If aux=False, return the true non-backtracking matrix, defined as the
    unnormalized transition matrix of a random walker that does not
    backtrack. If the graph has m edges, the NB-matrix is 2m x 2m. The rows
    and columns are ordered according to ordering (see half_incidence).

    If aux=True, return the auxiliary NB-matrix of a graph is the block
    matrix defined as

    B' = [0  D-I]
         [-I  A ]

    Where D is the degree-diagonal matrix, I is the identity matrix and A
    is the adjacency matrix. If the graph has n nodes, the auxiliary
    NB-matrix is 2n x 2n. The rows and columns are sorted in the order of
    the nodes in the graph object.

    Params
    ------

    graph (nx.Graph): the graph.

    aux (bool): whether to return the auxiliary or the true NB-matrix.

    ordering ('blocks' or 'consecutive'): ordering of the rows and columns
    if aux=False (see half_incidence). If aux=True, the rows and columns of
    the result will always be in accordance to the order of the nodes in
    the graph, regardless of the value of ordering.

    return_ordering (bool): if True, return the edge ordering used (see
    half_incidence).

    Returns
    -------

    matrix (scipy.sparse): (auxiliary) NB-matrix in sparse CSR format.

    matrix, ordering_func: if return_ordering=True.

    """
    if aux:
        degrees = graph.degree()
        degrees = sparse.diags([degrees[n] for n in graph.nodes()])
        ident = sparse.eye(graph.order())
        adj = nx.adjacency_matrix(graph)
        pseudo = sparse.bmat([[None, degrees - ident], [-ident, adj]])
        return pseudo.asformat('csr')

    else:
        # Compute the NB-matrix in a single pass on the non-zero elements
        # of the intermediate matrix.
        sources, targets, ord_func = half_incidence(
            graph, ordering, return_ordering=True)
        inter = np.dot(sources.T, targets).asformat('coo')
        inter_coords = set(zip(inter.row, inter.col))

        # h_coords contains the (row, col) coordinates of non-zero elements
        # in the NB-matrix
        h_coords = [(r, c) for r, c in inter_coords if (c, r) not in inter_coords]
        data = np.ones(len(h_coords))
        nbm = sparse.coo_matrix((data, list(zip(*h_coords))),
                                shape=(2*graph.size(), 2*graph.size()))

        # Return the correct format
        nbm = nbm.asformat('csr')
        return (nbm, ord_func) if return_ordering else nbm

def compute_mu(graph, val, vec):
    """Compute mu given the leading eigenpair.

    Params
    ------

    graph (nx.Graph): the graph.

    val (float): the leading eigenvalue.

    vec (np.array): the first half of the principal left unit eigenvector.

    Returns
    -------

    mu (float): a constant such that mu * vec is the 'correctly' normalized
    non-backtracking centrality.

    """
    degs = graph.degree()
    coef = sum(vec[n]**2 * degs(n) for n in graph)
    return np.sqrt(val * (val**2 - 1) / (1 - coef))

def nb_centrality(graph, normalized=True, return_eigenvalue=False, tol=0):
    
    """Return the non-backtracking centrality of each node.

    The nodes must be labeled by consecutive integers starting at zero.

    Params
    ------

    graph (nx.Graph): the graph.

    normalized (bool): whether to return the normalized version,
    corresponding to v^T P v = 1.

    return_eigenvalue (bool): whether to return the largest
    non-backtracking eigenvalue as part of the result.

    tol (float): the tolerance for eignevecto computation. tol=0 (default)
    means machine precision.

    Returns
    -------

    centralities (dict): dictionary of {node: centrality} items.

    centralities (dict), eigenvalue (float): if return_eigenvalue is True.

    """
    # Matrix computations require node labels to be consecutive integers,
    # so we need to (i) convert them if they are not, and (ii) preserve the
    # original labels as an attribute.
    graph = nx.convert_node_labels_to_integers(
        graph, label_attribute='original_label')

    # The centrality is given by the first entries of the principal left
    # eigenvector of the auxiliary NB-matrix
    val, vec = sparse.linalg.eigs(nb_matrix(graph, aux=True).T, k=1, tol=tol)
    val = val[0].real
    vec = vec[:graph.order()].ravel()

    # Sometimes the vector is returned with all negative components and we
    # need to flip the sign.  To check for the sign, we check the sign of
    # the sum, since every element has the same sign (or is zero).
    if vec.sum() < 0:
        vec *= -1

    # Currently, vec is unit length. The 'correct' normalization requires
    # that we scale it by \mu.
    if normalized:
        vec *= compute_mu(graph, val, vec)

    # Pack everything in a dict and return
    result = {graph.nodes[n]['original_label']: vec[n].real for n in graph}
    return (result, val) if return_eigenvalue else result

def DMPbetaC(G):
    '''
    critical point derived by DMP method

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    betac : float
        critical point.
    '''
    mu = 1
    [nb_vec,nb_val]= nb_centrality(G,normalized=False,return_eigenvalue=True)
    betac = mu/(nb_val-1+mu)
    
    return betac

def OneGbetaC(G):
    '''
    critical point derived by DMP method

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    betac : float
        critical point.
    '''
    mu = 1
    nbmatrix= nb_matrix(G)
    [eigval,eigvec] = sparse.linalg.eigs(nbmatrix)
    rho_B = max(abs(eigval))
    betac = mu/(rho_B-1+mu)
    
    return betac

def LS(nb_pev,avgk):
    '''
    calculate localization strenght $\mathcal{L}$

    Parameters
    ----------
    nb_pev : array
        primary eigenvector of nonbacktraing matrix.
    avgk : float
        mean degree of network.

    Returns
    -------
    ls : float
         Localization strenght $\mathcal{L}.

    '''
    y = abs(np.array(sorted(nb_pev.values())))
    ls = np.std(y)/(np.average(y)*avgk)

    return ls

def massDistribution(rhoSpreadBetac):
    '''
    calculate the mass distribution of rhos

    Parameters
    ----------
    rhoSpreadBetac : array
        rhos obatained by a large number of numerical simulation.

    Returns
    -------
    NormMd : dict
        distribution of rhos.

    '''
    md = {each:0 for each in sorted(set(rhoSpreadBetac))}
    for each in rhoSpreadBetac:
        md[each] = md[each]+1 
    
    NormMd = {rho:md[rho]/len(rhoSpreadBetac) for rho in md.keys()}
        
    return NormMd


 