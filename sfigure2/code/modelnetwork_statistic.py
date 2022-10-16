# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:49:05 2022

@author: Leyang Xue
"""
root_path  = 'F:/work/work4_dynamic' #please change the current path if run this code 

import sys
sys.path.append(root_path + '/InnovationDiffusion')
from utils import coupling_diffusion as cd
import networkx as nx 
import pandas as pd
import os

def network_statistic(loadpath,savepath):
    
    #load the network 
    networks = os.listdir(loadpath)
    #create the dict to save the result
    NetStats = {'network':[],'N':[],'L':[],'<k>':[],'r':[],'DMP_betac':[],'LS':[]}  
   
    for network in networks:
        
        #load the network
        G = nx.read_edgelist(loadpath+'/'+network)
        avgk = 2*G.size()/G.order() 
        
        #centralities of nonbacktracking matrix
        [nb_vec,nb_val]= cd.nb_centrality(G,normalized=False,return_eigenvalue=True)
        ls = cd.LS(nb_vec,avgk)
        betac_dmp = cd.DMPbetaC(G)
        
        #store the datasets
        NetStats['network'].append(network)
        NetStats['N'].append(G.order())
        NetStats['L'].append(G.size())
        NetStats['<k>'].append(avgk)
        NetStats['r'].append(nx.degree_assortativity_coefficient(G))
        NetStats['DMP_betac'].append(betac_dmp)
        NetStats['LS'].append(ls)
        
    Stats = pd.DataFrame(NetStats)
    Stats.to_csv(savepath+'/statistic_result.csv')

if __name__ == '__main__':
    
    root_path  = 'F:/work/work4_dynamic' #please change the current path if run this code 
    network_path = root_path + '/InnovationDiffusion/sfigure2/network'
    result_path = root_path + '/InnovationDiffusion/sfigure2/result'
    
    filenames = ['Assortativity', 'degreeExponent','density']
    
    for filename in filenames:
        loadpath = network_path + '/' + filename
        savepath = result_path + '/' + filename
        os.makedirs(savepath)
        network_statistic(loadpath,savepath)