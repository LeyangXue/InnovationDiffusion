# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:10:20 2022

@author: Leyang Xue
"""
root_path  = 'F:/work/work4_dynamic/' #please change the current path if run this code 

import sys
sys.path.append(root_path + '/InnovationDiffusion')
import os 
import pandas as pd
import pickle as pk
from utils import coupling_diffusion as cd
import numpy as np
import networkx as nx
from multiprocessing import Pool
import random

def Xulvi_Brunet_Sokolov(args):
    '''
    Xulvi_Brunet_Sokolov algorithms
    
    Parameters
    ----------
    G0 : graph
        original network.
    p  :float
       the ratio to swap the edges 
    assort : bool
       turning the network from current to assortative if True,
       otherwise, turning it to dissortative if False
    NetworkName : str
        the name of network.
    ResultPath : str
        the path to save the generated network.

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
        
    nx.write_edgelist(G,ResultPath+'/'+NetworkName+'.edgelist')       
    
def GenerateAssort(G,assort,ResultPath,ps,filename):
    '''
    generate the network with different magnititude of degree correlation
    
    Parameters
    ----------
    G : graph
        network.
    assort : bool
        turning the network from current to assortative if True,
        otherwise, dissortative if False.
    ResultPath : str
        the path to save the generated network..
    ps : array
        ratio of tuning the edegs.
    filename : str
        network name.

    Returns
    -------
    None.
    '''
    args = []
    for p in ps:
        if assort == True:
            #assort network
            NetworkName ='Assort_'+ 'p'+str(p)+'_'+filename
            args.append([G,p,assort,NetworkName,ResultPath])      
        elif assort == False:    
            #dissort network
            NetworkName = 'Dissort_'+'p'+str(p)+'_'+filename
            args.append([G,p,assort,NetworkName,ResultPath])

    pool = Pool(processes=5)
    pool.map(Xulvi_Brunet_Sokolov,args)  
    pool.close()
 
def network_statistic(network_path,result_path):
    
    networks_assort = os.listdir(network_path)
    
    for network_assort in networks_assort:
        #count the structural properities from each datasets
        networkname  = network_assort.split('.txt')[0]
        datapath = result_path+'/'+networkname
        networks_assort = os.listdir(datapath)
        NetStats = {'network':[],'N':[],'L':[],'<k>':[],'DMP_betac':[],'LS':[],'alphac_p':[]}  

        for network in networks_assort:
            
            #load the network
            G = nx.read_edgelist(network)
        
            #calculate the average degree
            avgk = 2*G.size()/G.order()
        
            #centralities of nonbacktracking matrix
            [nb_vec,nb_val]= cd.nb_centrality(G,normalized=False,return_eigenvalue=True)
            ls = cd.LS(nb_vec,avgk)
            betac_dmp = cd.DMPbetaC(nb_val)
        
            #store the datasets
            NetStats['network'].append(network_assort)
            NetStats['N'].append(G.order())
            NetStats['L'].append(G.size())
            NetStats['<k>'].append(avgk)
            NetStats['DMP_betac'].append(betac_dmp)
            NetStats['LS'].append(ls)
            NetStats['alphac_p'].append(2.63*ls)
        
        Stats = pd.DataFrame(NetStats)
        Stats.to_csv(result_path+'/'+networkname+'.csv')
    
if __name__ == '__main__':
    
    #set the path 
    root_path  = 'F:/work/work4_dynamic' #please change the current path if run this code 
    network_path = root_path + '/InnovationDiffusion/sfigure6/network'
    result_path = root_path + '/InnovationDiffusion/sfigure6/result'
    
    networks = ['soc-delicious.txt', 'soc-fb-pages-artist.txt','soc-advogato.txt','ca-InterdisPhysics.txt']
    #set the parameter
    assort = True
    dissort = False
    ps = np.arange(0.2,1.01,0.2)

    for network in networks:   
        
        print('network:', network)
        
        #load the networks
        data = np.loadtxt(network_path+'/'+network)
        G = cd.load_network(data)
    
        #adjust the networks
        #set the network name       
        filename = network.split('.txt')[0]
        resultpath = result_path + '/' + filename
        #create the resultpath
        cd.mkdir(resultpath)

        #generate assortative networks
        GenerateAssort(G,assort,resultpath,ps,filename)
        print('assort has finished')

        #generate dissortative networks
        GenerateAssort(G,dissort,resultpath,ps,filename)
        print('dissort has finished')


    #extract the networks in result path
    network_statistic(network_path,result_path)
    
    