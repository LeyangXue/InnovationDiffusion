# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:19:19 2022

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
import os 
import random 
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib as mpl   
from matplotlib.colors import ListedColormap 

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
    
def GenerateAssort(G,assort,ResultPath,ps,filename):
    '''
    generate the assortative or disassortative network based on the original network

    Parameters
    ----------
    G : graph 
        original network.
    assort : bool
        type of generated network, i.e assortative if ture, dissortative otherwise.
    ResultPath : str
        path to save the generated network.
    ps : array(float) 
        the ratio of swapped edge to turning the degree correlation. 
    filename : str
        filename to save the network.

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
            #disassort network
            NetworkName = 'Dissort_'+'p'+str(p)+'_'+filename
            args.append([G,p,assort,NetworkName,ResultPath])

    pool = Pool(processes=5) #run it with parallel computation
    pool.map(Xulvi_Brunet_Sokolov,args)  

def network_statistic(network_path,result_path):
    '''
    calculate the statistic properities of networks 

    Parameters
    ----------
    network_path: str
        path to load the network.
    result_path : str
        path to save the result.

    Returns
    -------
    None.

    '''
    networks_assort = os.listdir(network_path)
    NetStats = {'network':[],'N':[],'L':[],'<k>':[],'DMP_betac':[],'LS':[],'alphac_p':[]}  
    for network_assort in networks_assort:
        
        if '.edgelist' in network_assort:
            
            #load the network
            G = nx.read_edgelist(network_path+'/'+network_assort)
            
            #calculate the average degree
            avgk = 2*G.size()/G.order()
            
            #centralities of nonbacktracking matrix
            [nb_vec,nb_val]= cd.nb_centrality(G,normalized=False,return_eigenvalue=True)
            ls = cd.LS(nb_vec,avgk)
            betac_dmp = cd.DMPbetaC(G)
            
            #store the datasets
            NetStats['network'].append(network_assort)
            NetStats['N'].append(G.order())
            NetStats['L'].append(G.size())
            NetStats['<k>'].append(avgk)
            NetStats['DMP_betac'].append(betac_dmp)
            NetStats['LS'].append(ls)
            NetStats['alphac_p'].append(2.63*ls)
            
    Stats = pd.DataFrame(NetStats)
    Stats.to_csv(result_path+'/statistic_result.csv')

def RunSimuBetaC(args):
    '''
    run the numerical simulation with a given beta and alpha

    Parameters
    ----------
    args : list
        set of parameters.

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
    calculate the percentage of adopted individuals with betac_DMP probability
    
    Parameters
    ----------
    G : graph
        network.
    betac_DMP : float
        critical threshold derived by DMP.
    alphas : float
        intensity of conformity.
    file_path : str
        path to save the result.

    Returns
    -------
    None.

    '''
    simulation = 10000 #simulation times 
    mu = 1 #recovery probability
    
    for alpha in alphas:
        
        #with a given alpha
        args = []
        for i in np.arange(simulation): 
            infected = random.choice(list(G.nodes()))
            args.append([i,G,betac_DMP,mu,alpha,infected])
    
        pool = Pool(processes=30) #parallel computation   
        results_mc = pool.map(RunSimuBetaC,args)
        pk.dump(results_mc, open(file_path+'/'+str(alpha)+'_betac_mc.pkl', "wb"))

def PlotMassDistribution(alphas,path,file,figure):
    '''
    Plot the mass distribution of adopted individuals so that identify the tricritical point 

    Parameters
    ----------
    alphas : float
        intensity of conformity.
    path : str
        path to load the simulation results.
    file : str
        filename.
    figure : str
        figurename.
        
    Returns
    -------
    None.

    '''
    mass = {}
    color = plt.get_cmap('Paired')
    n = 4
    row = int(np.ceil(len(alphas)/n))
    fig, ax = plt.subplots(row,n,figsize=(n*2,row*2), sharey=True, sharex=True, constrained_layout=True)
    for i,alpha in enumerate(alphas):
        x = int(i/n)
        y = int(i%n)
        rhos_mc = cd.load(path+file+str(alpha)+'_betac_mc.pkl')
        mass[alpha]= cd.massDistribution(rhos_mc)
        ax[x,y].loglog(mass[alpha].keys(),mass[alpha].values(),'o',mec=color(1),mfc='white')
        ax[x,y].set_title(r'$\alpha$='+str(alpha))
        
    plt.savefig(path+file+figure,dpi=200)
    
def transformNetwork(network_path,result_path):
    '''
    Transform the network into the edgelist 
    
    Parameters
    ----------
    network_path : str
        path to load the network.
    result_path : str
        path to save the network.

    Returns
    -------
    None.

    '''
    G_assort = nx.read_edgelist(network_path+'/Assort_p1.0_G.edgelist')
    G_dissort = nx.read_edgelist(network_path+'/Dissort_p1.0_G.edgelist')

    Assort_egdelist = [[edge[0], edge[1]] for edge in nx.to_edgelist(G_assort)] 
    Dissort_edgelist = [[edge[0], edge[1]] for edge in nx.to_edgelist(G_dissort)] 

    np.savetxt(result_path+'/Assort_egdelist.csv', np.array(Assort_egdelist), delimiter=',', fmt='%s')
    np.savetxt(result_path+'/Dissort_egdelist.csv', np.array(Dissort_edgelist), delimiter=',', fmt='%s')

def nonbacktrackingCentrality(network_path,result_path):
    
    G_assort = nx.read_edgelist(network_path+'/Assort_p1.0_G.edgelist')
    G_dissort = nx.read_edgelist(network_path+'/Dissort_p1.0_G.edgelist')
    
    assort_nb = cd.nb_centrality(G_assort)
    dissort_nb = cd.nb_centrality(G_dissort)
    
    assort_nbpd = pd.Series(assort_nb)
    dissort_nbpd = pd.Series(dissort_nb)
    
    assort_nbpd.to_csv(result_path+'/assort_nb.csv')
    dissort_nbpd.to_csv(result_path+'/dissort_nb.csv')

def SpreadOnNetworks(result_path):
    '''
    Run the numerical simulation on assortative and dissortative networks
    and record the information of diffusion process
    
    Parameters
    ----------
    result_path : str
        path to save the result.

    Returns
    -------
    R_assort : list
        recovered node over t on assortative network.
    R_dissort : list
        recovered node over t on dissortative network.
    '''
    
    #set the parameter
    mu = 1
    alpha = 1
    infecteds = '4.0'
    assort_beatc = 0.10427817
    #assort_alphac = 1.17
    dissort_betac = 0.18253565
    #dissort_alphac = 0.58
    
    #load the network
    G_assort = nx.read_edgelist(network_path+'/Assort_p1.0_G.edgelist')
    G_dissort = nx.read_edgelist(network_path+'/Dissort_p1.0_G.edgelist')
    
    #spread
    [S_assort,I_assort, R_assort]= cd.SIR_G_df_Nodes(G_assort,assort_beatc,mu,alpha,infecteds, tmin = 0, tmax = float('Inf'))
    [S_dissort,I_dissort, R_dissort]= cd.SIR_G_df_Nodes(G_dissort,dissort_betac,mu,alpha,infecteds, tmin = 0, tmax = float('Inf'))
    
    #save the dataset
    pk.dump(S_assort, open(result_path + '/S_assort.pkl', "wb"))
    pk.dump(I_assort, open(result_path + '/I_assort.pkl', "wb"))
    pk.dump(R_assort, open(result_path + '/R_assort.pkl', "wb"))
    pk.dump(S_dissort, open(result_path + '/S_dissort.pkl', "wb"))
    pk.dump(I_dissort, open(result_path + '/I_dissort.pkl', "wb"))
    pk.dump(R_dissort, open(result_path + '/R_dissort.pkl', "wb"))
    
    return  R_assort, R_dissort

def InfectNodeAtrib(G,R,result_path,filename):
    '''
    Transform the adoption activity of each node over time into the node properities

    Parameters
    ----------
    G : graph
        network.
    R : list
        recovered nodes over time.
    result_path : str
        path to save the result.
    filename : str
        name to save the file.

    Returns
    -------
    None.

    '''
    t_activity = {}
    for t,nodes in enumerate(R):
        for node in nodes:
            t_activity[node] = t
    
    nodes_activity = {}
    for node in G.nodes():
        if t_activity.get(node) == None:
            nodes_activity[node] = -1 
        else:
            nodes_activity[node] = t_activity[node]
    
    Node_activity = pd.Series(nodes_activity)
    Node_activity.to_csv(result_path+'/'+filename+'.csv')    

def InfectEdgeAtrib(G,R,result_path,filename):
    
    #locate the spread path
    infect_path = {}
    for t,infecteds in enumerate(R[:-1]):
        for node in infecteds:
            for node_neighbors in nx.all_neighbors(G,node):
                if node_neighbors in R[t+1]:
                   infect_path[(node,node_neighbors)] = t
    
    #transform the spread_path to edge attribute
    edges = list(G.edges())
    interaction_attribute = pd.DataFrame(columns=['edges','infection_time'], index=range(len(edges)))
    infect_edges = list(infect_path.keys())
    for i,edge in enumerate(edges):
        interaction_attribute.iloc[i] =  edge[0] + ' (interacts with) ' + edge[1]
        if (edge in infect_edges) or ((edge[1],edge[0]) in infect_edges):            
            if edge in infect_edges:
                interaction_attribute.iloc[i]['infection_time'] = infect_path[edge]
            else:
                interaction_attribute.iloc[i]['infection_time'] = infect_path[(edge[1],edge[0])]
        else:
            interaction_attribute.iloc[i]['infection_time'] = -1
    
    interaction_attribute.to_csv(result_path+'/'+filename+'.csv')
    
def transformtoMatrix(R_assort):
    '''
    Transform the adoption activity of each node as a matrix format
    
    Parameters
    ----------
    R_assort : list
         recovered node over time.
         
    Returns
    -------
    None.

    '''
    ts_matirx_assort = np.zeros((10,int(max(R_assort.loc[:,'infected_nodes'])+1)))
    for inx in R_assort.index:
        each = R_assort.iloc[inx]
        if each['infected_nodes'] != -1:
            t = int(each['infected_nodes'])
            inx = int(each['nb']*10)
            ts_matirx_assort[inx,t] = ts_matirx_assort[inx,t]+1
    
    return  ts_matirx_assort
  
def PlotAxes(ax,xlabel,ylabel, title, mode=False):
    '''
    Decorate the axes

    Parameters
    ----------
    ax : axes
        axes.
    xlabel : str
        set the xlabel.
    ylabel : str
        set the ylabel.
    title : str
        set the title.
    mode : bool, optional
        whether to show the legend. The default is False.

    Returns
    -------
    None.

    '''
    fontsize = 35
    font_label = {'family': "Arial", 'size':fontsize}
    
    n_legend = 35
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='center',fontdict = {'family': "Arial", 'size':35}, pad=16)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    if mode == True:
        ax.legend(loc='upper left',bbox_to_anchor=(0.00,1.0), framealpha=0, fontsize=30)

def return_colorindex(p_assort):
    
    color_index = []
    for each in p_assort:
        if each != np.nan:
            if each != 1.0:
                color_index.append(int(each*10))
            else:
                color_index.append(9) 
            
    return  color_index

def plotcolorbar(font_label,figure_path,newcmap):
    '''    
    Plot a colorbar 
 
    Parameters
    ----------
    font_label : dict
        parameter of font label.
    figure_path : str
        path to plot the figure.
    newcmap:ListedColormap
        color map
    Returns
    -------
    None.

    '''
    #color = ["#D1E1E1","#C1D4D8","#B1C8CE","#A1BBC5","#91AFBB","#80A2B2","#7096A8","#60899F","#507D95","#40708C"]        
    fig, ax = plt.subplots(1,1, figsize=(2.8,8),constrained_layout=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    baraxes = mpl.colorbar.ColorbarBase(ax,cmap=newcmap,norm=norm,orientation='vertical',ticks=np.arange(0,1.01,0.1))
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=2,labelsize=35)
    baraxes.set_ticks(np.arange(0,1.01,0.2))
    baraxes.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
    baraxes.set_label('Proportion of recovered nodes ',fontdict=font_label)
    
    plt.savefig(figure_path+'/bar.png', dpi=300)
    plt.savefig(figure_path+'/bar.pdf')
    plt.savefig(figure_path+'/bar.eps')

def plotAssortNB(bins,result_path,figure_path,newcmap):
    '''
    plot the distribution of nonbacktracking centrality on assortative network
 
    Parameters
    ----------
    bins : TYPE
        DESCRIPTION.
    result_path : TYPE
        DESCRIPTION.
    figure_path : TYPE
        DESCRIPTION.
    newcmap : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    assort_nbpd = pd.read_csv(result_path+'/assort_nb.csv')
    [hist_assort,bins_edges_assort] = np.histogram(assort_nbpd,bins=bins)
    p_assort = Assort_Infected/hist_assort
    p_assort_index = return_colorindex(p_assort)
    color_assort = newcmap(p_assort_index)
    
    fig, ax = plt.subplots(1,1,figsize=(8,5),sharey= True, constrained_layout=True)  
    ax.bar(bins_edges_assort[:-1]+0.05,hist_assort,width=0.1,color = color_assort, log=True, edgecolor='white')
    PlotAxes(ax,'Non-backtracking centrality','Number of nodes', '', mode=False)
    ax.set_ylim(0.8,2000)
    ax.set_yticks([1,10,100,1000])
    ax.set_xticks(np.arange(0,1.01,0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.6, 10, r'$\mathcal{L}=0.33$', fontdict=font_label)
    
    plt.savefig(figure_path+'/assort_hist.png', dpi=300)
    plt.savefig(figure_path+'/assort_hist.pdf')
    plt.savefig(figure_path+'/assort_hist.eps')

def plotDissortNB(bins,result_path,figure_path,newcmap):
    
    dissort_nbpd = pd.read_csv(result_path+'/dissort_nb.csv')
    [hist_dissort,bins_edges_dissort] = np.histogram(dissort_nbpd,bins=bins)
    p_dissort = [infected/total for infected, total in zip(Dissort_Infected,hist_dissort) if total != 0]
    p_dissort_index = return_colorindex(p_dissort)
    color_dissort = newcmap(p_dissort_index)
    
    fig, ax = plt.subplots(1,1,figsize=(8,5), constrained_layout=True)  
    ax.bar(bins_edges_dissort[:-1]+0.05,hist_dissort,width=0.1,color = color_dissort, log=True, edgecolor='white')
    PlotAxes(ax,'Non-backtracking centrality','', '', mode=False)
    ax.set_ylim(0.8,2000)
    ax.set_yticks([1,10,100,1000])
    #ax.set_yticklabels(['','',''])

    ax.set_xticks(np.arange(0,1.01,0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.6, 10, r'$\mathcal{L}=0.11$', fontdict=font_label)

    plt.savefig(figure_path+'/dissort_hist.png', dpi=300)
    plt.savefig(figure_path+'/dissort_hist.pdf')
    plt.savefig(figure_path+'/dissort_hist.eps')
    
if __name__ == '__main__':
        
    #set the current path 
    network_path = root_path + '/InnovationDiffusion/figure2/network'
    figure_path = root_path + '/InnovationDiffusion/figure2/figure'
    result_path = root_path + '/InnovationDiffusion/figure2/result'
    
    #1 part: load and save the network 
    edgelist = np.loadtxt(network_path+'/ca-netscience.mtx')
    G = cd.load_network(edgelist)
    nx.write_edgelist(G,network_path+'/G_original.edgelist')
       
    #2 part: adjusting the network orienting different direction
    #set the parameter
    ps = np.arange(0.5,1.01,0.5) #the ratio of swaping edge     
    filename = 'G'
    
    #generate assortative networks
    assort = True
    GenerateAssort(G,assort,network_path,ps,filename)
    print('assort has finished')

    #generate dissortative networks
    dissort = False
    GenerateAssort(G,dissort,network_path,ps,filename)
    print('dissort has finished')
    
    #calculate the statistic properities of networks
    network_statistic(network_path,result_path)
    
    #3 part: calculate the real alphac
    #set the parameters
    alphas = list(map(lambda x: round(x,2),np.arange(0.8,1.2,0.02)))
    netname = 'Assort_p1.0_G.edgelist' # set the network as illustration example ('Assort_p1.0_G.edgelist','Dissort_p1.0_G.edgelist') 
    betac_dmp =  0.10427817  #betac_dmp from disassortative network: 0.182536
    
    file_path = network_path+'/'+netname.strip('.edgelist')
    cd.mkdir(file_path)
    
    #load the networks and spreading 
    G = nx.read_edgelist(network_path+'/'+netname) 
    RhoBetacSpread(G,betac_dmp,alphas,file_path)
    
    #plot the result of mass distribution
    file = '/Assort_p1.0_G/'
    figurename = "fig1.png"
    PlotMassDistribution(alphas, network_path,file,figurename)

    #4 part: transform the network into edgelsit, stored as the format .csv
    transformNetwork(network_path,result_path)
    
    #5 Part: calculate the nonbacktracking centrality of nodes 
    nonbacktrackingCentrality(network_path,result_path)
    
    #6 Part spreading on two networks
    [R_assort, R_dissort]= SpreadOnNetworks(result_path)
    
    #transform the infection datasets into node attribute
    G_assort = nx.read_edgelist(network_path+'/Assort_p1.0_G.edgelist')
    G_dissort = nx.read_edgelist(network_path+'/Dissort_p1.0_G.edgelist')
    
    #nodes 
    InfectNodeAtrib(G_assort,R_assort,result_path,'assort_activity')
    InfectNodeAtrib(G_dissort,R_dissort,result_path,'dissort_activity') 
    
    #edges
    InfectEdgeAtrib(G_assort,R_assort,result_path,'assort_edge_activity')
    InfectEdgeAtrib(G_dissort,R_dissort,result_path,'dissort_edge_activity')
    
    #7 part: load the adoption result over time 
    R_assort = pd.read_csv(result_path + '/assort_activity.csv')
    R_dissort = pd.read_csv(result_path + '/dissort_activity.csv')

    #transform the infection activity into the matrix
    Assort_matrix = transformtoMatrix(R_assort)
    Dissort_matrix = transformtoMatrix(R_dissort)
    
    #calculate the number of infected nodes
    Assort_Infected  = np.sum(Assort_matrix, axis=1)
    Dissort_Infected = np.sum(Dissort_matrix, axis=1)
    
    #8 part: plot the non-backtracking centrality distribution
    fontsize = 40
    font_label = {'family': "Arial", 'size':fontsize}
    color = ["#ABC9C8","#A0C3C5","#95BDC3","#89B7C0","#7EB1BD","#73AABB","#68A4B8","#5C9EB5","#5198B3","#4692B0"]
    newcmap = ListedColormap(color)  #create a new colormap

    #plot the colorbar
    plotcolorbar(font_label,figure_path,newcmap)
            
    #plot the distribution of nonbacktracking centrality on a assortative and dissortative network
    bg_color = "#c9c9dd" 
    bins = np.arange(0,1.01,0.1)
    plotAssortNB(bins,result_path,figure_path,newcmap)
    plotDissortNB(bins,result_path,figure_path,newcmap)
    
 

    


    
        