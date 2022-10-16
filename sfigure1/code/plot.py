# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:45:40 2022

@author: Leyang Xue
"""

import os 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def PlotAxes(ax,xlabel,ylabel, title, mode=False):
    
    fontsize = 28
    font_label = {'family': "Arial", 'size':fontsize}
    
    n_legend = 25
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = font_label,pad=10)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    if mode == True:
        ax.legend(loc='upper left',bbox_to_anchor=(0.00,1.0), framealpha=0, fontsize=30)
            
        
if __name__ == '__main__':
    
    root_path = 'F:/work/work4_dynamic' #need to change if run this code
    result_path = root_path + '/InnovationDiffusion/sfigure1/result'
    network_path =  root_path + '/InnovationDiffusion/sfigure1/network'
    figure_path = root_path + '/InnovationDiffusion/sfigure1/figure'
    
    #set the label
    netsname = [each.strip('.txt') for each in sorted(os.listdir(network_path))]
    labelname  = ['bio-dmela',
                'bio-yeast-protein-inter',
                'ca-AstroPh',
                'ca-AstroPhysTechObs',
                'ca-BayesianNet',
                'ca-CSphd',
                'ca-CondMat',
                'ca-Data-Mining',
                'ca-Database-System',
                'ca-FluidDynamics',
                'ca-GenTheoFiledParti',
                'ca-GrQc',
                'ca-HepPh',
                'ca-InterdisPhysics',
                'ca-NetSci2019',
                'ca-citeseer',
                'cit-DBLP',
                'cit-HepPh',
                'cit-HepTh',
                'email-EU',
                'email-EuAll',
                'email-dnc',
                'email-enron-large',
                'soc-Slashdot0811',
                'soc-academia',
                'soc-advogato',
                'soc-delicious',
                'soc-fb-Huawei',
                'soc-fb-pages-artist',
                'soc-fb-pages-company',
                'soc-fb-pages-government',
                'soc-fb-pages-media',
                'soc-musae-git',
                'tech-as-caida2007',
                'tech-p2p-gnutella',
                'tech-pgp',
                'web-EPA',
                'web-webbase-2001']
           
    #set the parameter
    fontsize = 28
    font_label = {'family': "Arial", 'size':fontsize}
    color = ["#4063a3",'#cfcfcf','#ddc000','#d7aca1',"#34b6c6","#79ad41",'#3b3a3e']

    #load the data
    results = pd.read_csv(result_path+'/summary_results.csv')
    index = list(map(lambda x:round(x,2),np.arange(0,1.01,0.05)))
    betac_name = pd.DataFrame(index=index,columns=netsname)
    for name in netsname:
        result = results[results['network']==name]
        betac_name[name] = list(result['betac_mc'] - result['betac_dmp'])
    
    #plot the figure
    fig, ax = plt.subplots(1,1,figsize=(16,8), constrained_layout = True)
    vp = ax.violinplot(betac_name, np.arange(0,len(netsname),1),widths=0.6,showmeans=False, showmedians=False, showextrema=True)
    ax.set_ylim(-0.2,0.2)
    ax.set_xlim(-0.5,37.5)
    ax.set_xticks(np.arange(0,len(netsname),1))
    ax.set_xticklabels(labelname, rotation=90)
    ax.grid(axis='both')
    PlotAxes(ax,'Networks',' \n', '')
    fig.text(0.01,0.35,'Numerical '+ r'$\beta_c$ ' +'- Theoretical '+ r'$\beta_c$',rotation=90, fontdict=font_label)
    # styling:
    for body in vp['bodies']:
        body.set_alpha(0.35)
        #body.set_facecolor(color[4])
        body.set_edgecolor('white')
        
    plt.savefig(figure_path+'/criticalPoint.png', dpi=300)
    plt.savefig(figure_path+'/criticalPoint.eps')
    plt.savefig(figure_path+'/criticalPoint.pdf')