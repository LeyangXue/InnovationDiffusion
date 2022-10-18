# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:33:07 2022

@author: Leyang Xue
"""
root_path = 'F:/work/work4_dynamic' #change the current path if run this code

import sys
sys.path.append(root_path+'/InnovationDiffusion')
from utils import coupling_diffusion as cd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, spearmanr

def LR(x,y):
    
    lr = LinearRegression()
    x_values = np.reshape(np.array(x),(len(x),1))
    y_values = np.reshape(np.array(y),(len(y),1))
    lr.fit(x_values,y_values)
    beta1 = lr.coef_[0][0]
    beta0 = lr.intercept_[0]
    #predict = np.exp(beta0)*np.power(x,beta1)#lmb0,lmb1
    
    #print('beta1',beta0)
    #print('beta0',beta1)
    
    return beta0,beta1

def combinedata(statistic,ResultPath):

    rhos_pd = {'network':[], '0rhos':[], '0rhos_CI':[], '2rhos':[], '2rhos_CI':[],'5rhos':[], '5rhos_CI':[],'10rhos':[], '10rhos_CI':[],
               '0time':[], '0time_CI':[],'2time':[], '2time_CI':[], '5time':[], '5time_CI':[],'10time':[], '10time_CI':[]}
    nmapname = {0:'0',1:'2',2:'5',3:'10'}
    
    #create the file name and load the data
    networkname  = statistic['network']
    rhos_name = [name+'_results_mc.pkl' for name in networkname]
    t_name = [name+'_results_t.pkl' for name in networkname]
        
    for tname, rhoname in zip(t_name,rhos_name):
        
        #load the spread data 
        t_data =  cd.load(ResultPath + '/' + tname)
        rho_data = cd.load(ResultPath + '/' + rhoname)
        
        resultnames = tname.split('_results')
        rhos_pd['network'].append(resultnames[0])
        
        for i, betac in enumerate(sorted(rho_data.keys())):    
            
            #store the name
            inx = nmapname[i]
            
            #calculate the average and CI of rhos for a given betac
            rho_avg = np.average(rho_data[betac])
            rho_ci = st.norm.interval(alpha=0.95, loc=rho_avg, scale=st.sem(rho_data[betac]))
            rho_ci_value = rho_avg-rho_ci[0]
            
            #calculate the average and CI of t 
            time_avg = np.average(t_data[betac])
            time_ci = st.norm.interval(alpha=0.95, loc=time_avg, scale=st.sem(t_data[betac])) 
            time_ci_value = time_avg - time_ci[0]
            
            rhos_pd[inx + 'time'].append(time_avg)
            rhos_pd[inx + 'time_CI'].append(time_ci_value)
            rhos_pd[inx + 'rhos'].append(rho_avg)
            rhos_pd[inx + 'rhos_CI'].append(rho_ci_value)
        
    networks_rhos = pd.DataFrame(data=rhos_pd)
    merged_rhos = pd.merge(statistic,networks_rhos, on='network')
    merged_rhos.to_csv(ResultPath + '/rhos_results.csv')
    
def PlotAxes(ax,xlabel,ylabel, title, mode=False):
    
    fontsize = 24
    font_label = {'family': "Arial", 'size':fontsize}
    
    n_legend = 22
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = font_label,pad=10)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    if mode == True:
        ax.legend(loc='upper left',bbox_to_anchor=(0.00,1.0), framealpha=0, fontsize=30)
            
if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #change the current path if run this code
    ResultPath = root_path + '/InnovationDiffusion/sfigure3/result'
    figurePath = root_path + '/InnovationDiffusion/sfigure3/figure'

    #load the statistic result
    statistic = pd.read_csv(ResultPath+'/complete_datasets_updated.csv')
        
    #calculate the average result for each network
    combinedata(statistic,ResultPath)             
        
    #load the results
    results = pd.read_csv(ResultPath+'/rhos_results.csv')
    alphacr = results.sort_values(by = ['R_alphaC'])
    
    #log fit
    #[beta0,beta1]= LR(np.log(alphacr['R_alphaC']),alphacr['10rhos'])
    #predict = beta1*np.log(alphacr['R_alphaC']) + beta0
    
    #linear fit
    #rhos
    #parameter_rhos = np.polyfit(np.log(alphacr['R_alphaC']),alphacr['10rhos'],2)
    #model_rhos= np.polyval(parameter_rhos, np.log(alphacr['R_alphaC']))
    
    #time 
    #parameter_time = np.polyfit(np.log(alphacr['R_alphaC']),alphacr['10time'],2)
    #model_time = np.polyval(parameter_time, np.log(alphacr['R_alphaC']))
    
    #kendalltau correlation
    #[ktau_rho_r,ktau_rho_p] = kendalltau(alphacr['R_alphaC'],alphacr['10rhos'])
    #[ktau_time_r,ktau_time_p] = kendalltau(alphacr['R_alphaC'],alphacr['10time'])
    
    #spearman correlation
    [s_rho_r,s_rho_p] = spearmanr(alphacr['R_alphaC'],alphacr['10rhos'])
    [s_time_r,s_time_p] = spearmanr(alphacr['R_alphaC'],alphacr['10time'])

    #plot the figure
    ms = 200
    mew = 1.5
    bg_color = 'black'#'#ababab' #'#CFD2CF' #3b3a3e
    color = ["#4063a3",'#cfcfcf','#ddc000','#d7aca1',"#34b6c6","#79ad41",'#3b3a3e']
    fontsize = 24
    font_label = {'family': "Arial", 'size':fontsize}
    alpha = 1
    
    fig, ax = plt.subplots(1,2,figsize=(13,6), sharex=True,constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.2,
        hspace=0.0, wspace=0.0)
    
    ax[0].scatter(alphacr['R_alphaC'], alphacr['10rhos'],marker='o',s=ms, edgecolors = 'white', linewidths=mew, color=color[4], alpha=alpha)
    ax[0].text(0.005,0.2, r'$\rho$ = '+str(round(s_rho_r,2)), fontdict=font_label)
    ax[0].text(0.006,0.08, '(***)', fontdict= {'family': "Arial", 'size':24, "color":bg_color})

    #ax[0].plot(alphacr['R_alphaC'],model_rhos, color = color[-1], lw=1.0, ls='solid')
    #ax[0].errorbar(alphacr['R_alphaC'],alphacr['10rhos'],alphacr['10rhos_CI'], ecolor=color[0],elinewidth=mew, color= color[0], lw='')
    PlotAxes(ax[0],' ','Proportion of recovered nodes \n'+r'$R(\infty)/N$', '(a)',mode=False)#
    ax[0].set_xscale('log')
    ax[1].scatter(alphacr['R_alphaC'], alphacr['10time'],marker='o',s=ms, edgecolors = 'white', linewidths=mew, color=color[5],alpha=alpha)
    ax[1].text(0.005,13.2, r'$\rho$ = '+str(round(s_time_r,2)), fontdict=font_label)
    ax[1].text(0.006,7.5, '(***)', fontdict= {'family': "Arial", 'size':24, "color":bg_color})

    #ax[1].plot(alphacr['R_alphaC'],model_time, color = color[-1], lw=1.0,ls='solid')
    PlotAxes(ax[1],' ','The time \n'+'to reach the 10% adoption', '(b)',mode=False)#
    fig.text(0.45,0.02,'Numerical '+r'$\alpha_c$',fontdict = font_label)
    
    plt.savefig(figurePath+'/TricriticalPoint.png', dpi=300)
    plt.savefig(figurePath+'/TricriticalPoint.eps')
    plt.savefig(figurePath+'/TricriticalPoint.pdf')
