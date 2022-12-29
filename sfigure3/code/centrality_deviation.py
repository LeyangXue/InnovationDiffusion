# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:53:41 2022

@author: Leyang Xue
"""

root_path  = 'G:/work/work4_dynamic' #please change the current path if run this code 

import sys
sys.path.append(root_path + '/InnovationDiffusion')
from utils import coupling_diffusion as cd 
import pandas as pd
import os 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_log_error

def Gini(y):
    '''
    calculate the Gini coefficient for a given centrality 

    Parameters
    ----------
    y : array
        value of centrality.

    Returns
    -------
    gini : float
        gini coefficient.
    '''
    
    y = np.sort(y)
    y = y / sum(y)
    y = np.cumsum(y)
    gini= 1 - 2*sum(y/(len(y)+1))
    return gini

def CV(y,avgk):
    '''
    calculate the coefficient of variation for a given centrality 

    Parameters
    ----------
    y : array
        value of centrality.
    avgk : float
        mean degree.

    Returns
    -------
    ls : float
        localization strength, i,e, \matchal{L}.

    '''
    ls = np.std(y)/(np.average(y)*avgk)
    
    return ls

def IPR4(y):
    '''
    calculate the inversal partition rate for a given centrality

    Parameters
    ----------
    y : array
        value of centrality.

    Returns
    -------
    value: float
         inversal partition rate.

    '''
    return np.sum(np.power(y,4))

def Eigen(G):
    '''
    calculate the eigenvector centrality of network for a given centrality

    Parameters
    ----------
    G : graph 
        network.

    Returns
    -------
    eigencentrality : list
        the value of eigenvector centrality.

    '''
    eigenvector = nx.eigenvector_centrality(G,max_iter=1000)
    eigencentrality = list(eigenvector.values())
    
    return eigencentrality

def nonbacktracking(G):
    '''
    calculate the nonbacktracking centrality centrality of network

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    nonbackcentrality : list
        float.
    '''
    nonback = cd.nb_centrality(G, normalized=True, return_eigenvalue=False, tol=0)
    nonbackcentrality = list(nonback.values())
    
    return nonbackcentrality

def DeviationModelNetwork(networkpath,resultpath,statistic):
    '''
    calculate the deviation of centrality on model networks using different metrics,
    i.e. Gini, IPR, CV
    
    Parameters
    ----------
    networkpath : str
        path to save the network.
    resultpath : str
        path to save the network.
    statistic : array
        basic statistic information of networks.

    Returns
    -------
    None.

    '''
    #create the dict to save the data 
    centrality_pd = {'name':[],'Gini_eigen':[],'Gini_nonback':[],'CV_eigen':[],'CV_nonback':[],'IPV4_eigen':[],'IPV4_nonback':[]}  
    
    #load the network so that we can calculate the structural properities
    networks = os.listdir(networkpath)
    for network in networks:
        
        #each network
        print('networks:', network) 
        #store the name
        centrality_pd['name'].append(network)
        
        #1. load the network
        G = nx.read_edgelist(networkpath + '/'+ network)
        
        #2. calculate the eigenvector centrality
        eigenc = Eigen(G)
        
        #calculate the deviation by gini, IPR and CV
        gini_eigen =  Gini(eigenc)
        ipr_eigen = IPR4(eigenc)
        avgk = 2*G.size()/G.order()
        cv_eigen = CV(eigenc,avgk)
        
        #add metrics into the dataframe
        centrality_pd['Gini_eigen'].append(gini_eigen)
        centrality_pd['CV_eigen'].append(cv_eigen)
        centrality_pd['IPV4_eigen'].append(ipr_eigen)
        
        #3. calculate the nonbackcentrality centrality
        nonbackc = nonbacktracking(G)
        
        #calculate the deviation by gini, IPR4 and CV
        gini_nonback =  Gini(nonbackc)
        ipr_nonback = IPR4(nonbackc)
        cv_nonback = CV(nonbackc,avgk)
        
        #add metrics into the dataframe
        centrality_pd['Gini_nonback'].append(gini_nonback)
        centrality_pd['CV_nonback'].append(cv_nonback)
        centrality_pd['IPV4_nonback'].append(ipr_nonback)
        
    centrality = pd.DataFrame(centrality_pd)
    centrality.to_csv(resultpath+'/deviationModelNetwork.csv')
    
    merged_data = pd.merge(statistic,centrality, on='name')  
    merged_data.to_csv(resultpath+'/summaryModelNetwork.csv')

def DeviationRealNetwork(networkpath,resultpath,statistic):
    '''
    calculate the deviation of centrality on real-world networks using different metrics,
    i.e. Gini, IPR, CV
    
    Parameters
    ----------
    networkpath : str
        path to save the network.
    resultpath : str
        path to save the network.
    statistic : array
        basic statistic information of networks.

    Returns
    -------
    None.

    '''
    centrality_pd = {'name':[],'Gini_eigen':[],'Gini_nonback':[],'CV_eigen':[],'CV_nonback':[],'IPV4_eigen':[],'IPV4_nonback':[]}  
    
    networks = os.listdir(networkpath)
    for network in networks:
        
        name = network.strip('.txt')
        print('networks:', name) 
        
        #store the name
        centrality_pd['name'].append(name)
        
        #1. load the network
        datasets = np.loadtxt(networkpath+'/'+network, delimiter =',')
        G = cd.load_network(datasets)
        
        #2. calculate the eigenvector centrality
        eigenc = Eigen(G)
        
        #calculate the deviation by gini, IPR4 and CV
        gini_eigen =  Gini(eigenc)
        ipr_eigen = IPR4(eigenc)
        avgk = 2*G.size()/G.order()
        cv_eigen = CV(eigenc,avgk)
        
        #add metrics into the dataframe
        centrality_pd['Gini_eigen'].append(gini_eigen)
        centrality_pd['CV_eigen'].append(cv_eigen)
        centrality_pd['IPV4_eigen'].append(ipr_eigen)
        
        #3. calculate the nonbackcentrality centrality
        nonbackc = nonbacktracking(G)
        
        #calculate the deviation by gini, IPR4 and CV
        gini_nonback =  Gini(nonbackc)
        ipr_nonback = IPR4(nonbackc)
        cv_nonback = CV(nonbackc,avgk)
        
        #add metrics into the dataframe
        centrality_pd['Gini_nonback'].append(gini_nonback)
        centrality_pd['CV_nonback'].append(cv_nonback)
        centrality_pd['IPV4_nonback'].append(ipr_nonback)
        
    centrality = pd.DataFrame(centrality_pd)
    centrality.to_csv(resultpath+'/deviationRealNetwork.csv')
    
    merged_data = pd.merge(statistic,centrality, on='name')  
    merged_data.to_csv(statistic+'/summaryRealNetwork.csv')

def SpearmanCorrelation(modelresults):
    '''
    calculate the spearman correlation between alphac and metrics

    Parameters
    ----------
    modelresults : dataframe
        data with different metrics.

    Returns
    -------
    model_corr : dict
        spearman correlation.
    model_p : dict
        p-value.

    '''
    index_name = ['Gini_eigen','Gini_nonback','CV_eigen','CV_nonback','IPV4_eigen','IPV4_nonback']
    model_corr = {}
    model_p = {}
    for indexname in index_name:
        [rho,p]= spearmanr(modelresults['alphac'], modelresults[indexname])
        model_corr[indexname] = rho
        model_p[indexname] = p
    
    return model_corr, model_p 

def MetricsAccuracy(modelresults,realresults):
    '''
    compare the MSLE among different metircs about the deviation of centrality

    Parameters
    ----------
    modelresults : dataframe
    data with different metrics.

    Returns
    -------
    powerlawmodel_MSLE : dict
        MSLE.
    predictreal_alphac : dict
        predicted alphac.

    '''
    index_name = ['Gini_eigen','Gini_nonback','CV_eigen','CV_nonback','IPV4_eigen','IPV4_nonback']
    powerlawmodel_parameter = {}
    powerlawmodel_MSLE = {}
    predictreal_alphac = {}
    spearmancorr = {}
    
    for indexname in index_name:
        parameter_rhos = np.polyfit(np.log(modelresults[indexname]),np.log(modelresults['alphac']),1)
        powerlawmodel_parameter[indexname] = parameter_rhos
        predicted_alphac = np.exp(parameter_rhos[1])*np.power(realresults[indexname],parameter_rhos[0])
        MSLE = mean_squared_log_error(realresults['R_alphaC'],predicted_alphac)
        powerlawmodel_MSLE[indexname] = MSLE
        [rho,p]= spearmanr(realresults['R_alphaC'], predicted_alphac)
        spearmancorr[indexname] =(rho,p)
        predictreal_alphac[indexname] = [predicted_alphac,realresults['R_alphaC']]
    
    return powerlawmodel_MSLE, spearmancorr, predictreal_alphac

def PlotAxes(ax,xlabel,ylabel,title='',legends=False):
    
    font_label = {'family': "Arial", 'size':22}
    n_legend = 20
    bg_color = '#ababab' #'#CFD2CF' #3b3a3e
    ax.set_xlabel(xlabel, fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label,color=bg_color)
    ax.set_title(title, loc='center',fontdict =font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    if legends == True:
        ax.legend(loc='upper right',bbox_to_anchor=(1.0,1.0), framealpha=0, fontsize=n_legend)
    #ax.locator_params(tight='True')
    #ax.ticklabel_format(style='sci')

def PlotHist(model_corr,powerlawmodel_MSLE):
        
    #process the dataset
    #corr_x = list(model_corr.keys())
    corr_y = list(model_corr.values())
    eigenvector = list(map(lambda x:round(x,2),[corr_y[0],corr_y[2],corr_y[4]]))
    nonbacktrack = list(map(lambda x:round(x,2),[corr_y[1],corr_y[3],corr_y[5]]))
    
    #MSLE_x = list(powerlawmodel_MSLE.keys())
    MSLE_y = list(powerlawmodel_MSLE.values())
    eigenvector_msle = list(map(lambda x:round(x,2), [MSLE_y[0],MSLE_y[2],MSLE_y[4]]))
    nonbacktrack_msle = list(map(lambda x:round(x,2),[MSLE_y[1],MSLE_y[3],MSLE_y[5]])) 
    
    bg_color = 'white'#'#3b3a3e'#CFD2CF
    N = 3
    width = 0.28
    lw = 0.8
    colors = ["#ddc000","#79ad41","#34b6c6","#4063a3"]
    labels = ['Gini', 'CV/'+r'$\langle k \rangle$', 'IPR']
    tsize = 20 
    
    fig,ax = plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
    
    #axes0
    ind = np.arange(N) 
    ax0_eig =ax[0].bar(ind,eigenvector,width,bottom=0,align='edge',label = 'Eigenvector',edgecolor = bg_color, color = colors[0],linewidth=lw,alpha = 0.85)
    ax0_non = ax[0].bar(ind+width,nonbacktrack,width,bottom=0,align='edge',label = 'Non-backtracking',edgecolor = bg_color, color = colors[1],linewidth=lw, alpha = 0.85)
    
    ax[0].set_xticks(ind+width)
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks([0.0,0.5,1.0,1.5])
    ax[0].set_ylim(-0.3,1.6)
    ax[0].axhline(0, color='grey', linewidth=0.8)
    
    #ax[0].grid(axis="y")
    PlotAxes(ax[0],'metrics','Spearman rank correlation'+r', $\rho$',title='(a)', legends=True)
    ax[0].bar_label(ax0_eig,padding=3,size=tsize)#label_type='center'
    ax[0].bar_label(ax0_non,padding=3, size=tsize)#,label_type='center'

    #axes1 
    ax1_eig = ax[1].bar(ind,eigenvector_msle,width,bottom=0,align='edge',label = 'Eigenvector',edgecolor = bg_color, color = colors[0],linewidth=lw,alpha = 0.85)
    ax1_non = ax[1].bar(ind+width,nonbacktrack_msle,width,bottom=0,align='edge',label = 'Non-backtracking',edgecolor = bg_color, color = colors[1],linewidth=lw, alpha = 0.85)
    
    ax[1].set_xticks(ind+width)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0,0.9)
    
    PlotAxes(ax[1],'metrics','MSLE',title='(b)',legends=False)
    ax[1].bar_label(ax1_eig,padding=3, size=tsize)#label_type='center'
    ax[1].bar_label(ax1_non,padding=3, size=tsize)
    
    plt.savefig(figure_path+'/metrics.eps',dpi = 300)
    plt.savefig(figure_path+'/metrics.png')
    plt.savefig(figure_path+'/metrics.pdf')

def PlotMetricComparision(predictreal_alphac,spearmancorr):

    font_label = {'family': "Arial", 'size':22}
    bg_color = '#ababab' #'#CFD2CF' #3b3a3e
    color = ["#34b6c6","#79ad41",'#ddc000','#1f77b4','#ff7f0e','#cfcfcf',"#4063a3"]
    
    mec = 'white' 
    s = 12
    mew = 1.2
    alpha = 1.0
    x = np.arange(8*np.power(1/10,7),200,0.1)
    fig,ax = plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
    
    ax[0,0].loglog(predictreal_alphac['Gini_eigen'][0],predictreal_alphac['Gini_eigen'][1],marker='o', mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[0])
    ax[0,0].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[0,0].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['Gini_eigen'][0],3))+'***',fontdict=font_label)
    ax[0,0].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['Gini_eigen'],3)),fontdict=font_label)

    ax[1,0].loglog(predictreal_alphac['Gini_nonback'][0],predictreal_alphac['Gini_nonback'][1],marker='s',mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[0])
    ax[1,0].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[1,0].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['Gini_nonback'][0],3))+'***',fontdict=font_label)
    ax[1,0].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['Gini_nonback'],3)),fontdict=font_label)

    ax[0,1].loglog(predictreal_alphac['IPV4_eigen'][0],predictreal_alphac['IPV4_eigen'][1],marker='o',mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[1])
    ax[0,1].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[0,1].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['IPV4_eigen'][0],3))+'***',fontdict=font_label)
    ax[0,1].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['IPV4_eigen'],3)),fontdict=font_label)

    ax[1,1].loglog(predictreal_alphac['IPV4_nonback'][0],predictreal_alphac['IPV4_nonback'][1],marker='s',mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[1])
    ax[1,1].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[1,1].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['IPV4_nonback'][0],3))+'*',fontdict=font_label)
    ax[1,1].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['IPV4_nonback'],3)),fontdict=font_label)

    ax[0,2].loglog(predictreal_alphac['CV_eigen'][0],predictreal_alphac['CV_eigen'][1],marker='o',mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[2])
    ax[0,2].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[0,2].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['CV_eigen'][0],3))+'***',fontdict=font_label)
    ax[0,2].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['CV_eigen'],3)),fontdict=font_label)

    ax[1,2].loglog(predictreal_alphac['CV_nonback'][0],predictreal_alphac['CV_nonback'][1],marker='s',mec=mec,ms=s, ls='', alpha=alpha, mew = mew, color = color[2])
    ax[1,2].plot(x, x, ls='solid',color = bg_color, lw=1.2)
    ax[1,2].text(0.0015,0.000015, r'$\rho$ ='+ str(round(spearmancorr['CV_nonback'][0],3))+'***',fontdict={'family': "Arial", 'size':22})
    ax[1,2].text(0.001,0.000001, 'MSLE='+ str(round(powerlawmodel_MSLE['CV_nonback'],3)),fontdict={'family': "Arial", 'size':22, 'weight': 'bold'})

    PlotAxes(ax[0,0],'','  \n Eigenvector',title='(a) Gini',legends=False)
    PlotAxes(ax[1,0],' \n ','  \n Non-backtracking',title='',legends=False)
    PlotAxes(ax[0,1],'','',title='(b) IPR',legends=False)
    PlotAxes(ax[1,1],' \n ','',title='',legends=False)
    PlotAxes(ax[0,2],'','',title='(c) RSD',legends=False)
    PlotAxes(ax[1,2],' \n ','',title='',legends=False)
    
    fig.text(0, 0.45, 'Numerical '+ r'$\alpha_c$', fontdict = font_label, rotation = 'vertical')
    fig.text(0.48, 0.03, 'Predicted '+ r'$\alpha_c$', fontdict = font_label)
    
    ax[0,0].set_yticks([0.0001,0.1, 100])
    ax[1,0].set_xticks([0.0001,0.1, 100])

    plt.savefig(figure_path+'/MSLE.eps',dpi = 300)
    plt.savefig(figure_path+'/MSLE.png')
    plt.savefig(figure_path+'/MSLE.pdf')
       
       
if __name__ == '__main__':
    
    root_path  = 'G:/work/work4_dynamic' #please change the current path if run this code 

    modelnetworkpath = root_path + '/InnovationDiffusion/sfigure3/network/modelnetworks'
    realnetworkpath = root_path + '/InnovationDiffusion/sfigure3/network/realnetworks'
    resultpath = root_path + '/InnovationDiffusion/sfigure3/result'
    figure_path = root_path + '/InnovationDiffusion/sfigure3/figure'
    
    #load the data from the model network
    modelstatistic = pd.read_csv(resultpath+'/modelnetwork_statistic/Results.csv')
    DeviationModelNetwork(modelnetworkpath,resultpath,modelstatistic)
        
    realstatistic = pd.read_csv(resultpath + '/realnetwork_statistic/complete_datasets.csv')
    DeviationRealNetwork(realnetworkpath,resultpath,realstatistic)
    
    #load the results from network model and realnetworks
    modelresults = pd.read_csv(resultpath+'/summaryModelNetwork.csv')
    realresults =  pd.read_csv(resultpath+'/summaryRealNetwork.csv')
    
    #calculate the spearmanr correlation on model networks 
    [model_corr, model_p]= SpearmanCorrelation(modelresults)

    #fit the powerlaw model
    [powerlawmodel_MSLE, spearmancorr, predictreal_alphac] = MetricsAccuracy(modelresults,realresults)
    
    #plot the result 
    PlotMetricComparision(predictreal_alphac, spearmancorr)