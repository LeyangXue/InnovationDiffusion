# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:04:32 2022

@author: Leyang Xue
"""

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd

def theoretical_results(alphac, ks, lambd):
   '''
   calculate the betac for a given alphac on uncorrelated network  

   Parameters
   ----------
   alphac : float
       tricritical point.
   ks : array
       degree of network.
   lambd : float
       constant of localization phenonmenons.
    
    Returns
    -------
    alphac : float
        tricritical point.
    betac : float
        critical point.

    '''
   betac = np.zeros((len(alphac),len(ks)))
   for i,k in enumerate(ks):
       betac[:,i] = 1/(np.power(k,3)*np.power(alphac/lambd,2)+k-1)
       
   return alphac,betac     

def classfiy_density(data):
   '''
   classify the networks into the group according to mean degree of networks 

    Parameters
    ----------
    data : dataframe
        datasets from 74 real networks.

    Returns
    -------
    data : dataframe
        classified the network into several groups according to mean degree.

    ''' 
   intk = []
   for i in data.index:
       if data.iloc[i]['k'] < 5:
          intk.append(r'$\langle k\rangle \in[2,5)$')
       elif data.iloc[i]['k'] < 10:
          intk.append(r'$\langle k\rangle \in[5,10)$')
       elif data.iloc[i]['k'] < 20:
          intk.append(r'$\langle k\rangle \in[10,20)$') 
       elif data.iloc[i]['k'] < 100:
          intk.append(r'$\langle k\rangle \in[20,100)$')
       else:
          intk.append(r'$\langle k\rangle \in[100,\infty)$')
   
   data['group_k'] = pd.Series(intk)    
   
   return data

def PlotAxes(ax,xlabel,ylabel,title):
    
    fontsize = 38
    n_legend = 36
    
    font_label = {'family': "Arial", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    ax.set_title(title, loc='left',fontdict = {'family': "Arial", 'size':45})
    #ax.text(index[0],index[1],title,fontdict = font_label)
    
def plot_alpbacBetac(axes,complete_data,alphac,betac,color):
    '''
    plot the efficience of diffusion for different kinds of networks
    
    Parameters
    ----------
    axes : axes
        axes.
    complete_data : dataframe
        datasets from 66 real networks.
    alphac : float
        tricritical point.
    betac : float
        critical point.
    color : str
        color.

    Returns
    -------
    None.

    '''
    size = 350
    linecolor = '#3b3a3e'
    n_legend = 24
   
    axes.plot(alphac,betac[:,0],'-',color=linecolor, label=r'$\langle k\rangle$=2',lw=1.5)
    sns.scatterplot(data=complete_data, x='R_alphaC', y= 'MC_betac',hue='Type',style='Type',s=size,ax=axes,legend="full",palette=color)
    PlotAxes(axes,' ',' ','(a)')#x:Predicted  '+ r'$\alpha_c$, y:Theoretical  '+r'$\beta_c$
   
    axes.legend(loc='upper right',bbox_to_anchor=(1.0,1.0),framealpha=0, fontsize=n_legend,handletextpad=0.1,markerscale=3.0)

def plot_alpbacBetac_loglog(axes,data,alphc,betac,color):
   '''
    Plot the efficiencey of networks over different average degree 

    Parameters
    ----------
    axes : axes
        axes.
    data : dataframe
        dataframe.
    alphc : float
        tricritical point.
    betac : float
        critical point.
    color : str
        color.

    Returns
    -------
    None.

    ''' 
   #set the parameter of plot 
   orders = [r'$\langle k\rangle \in[2,5)$',r'$\langle k\rangle \in[5,10)$',r'$\langle k\rangle \in[10,20)$',r'$\langle k\rangle \in[20,100)$',r'$\langle k\rangle \in[100,\infty)$']
   size = 350
   n_legend = 24
   lw = 2
   alpha2 = 1
   
   #plot the figure
   sns.scatterplot(data=data, x='R_alphaC', y= 'MC_betac', hue='group_k',style='group_k',hue_order=orders,style_order=orders,legend="full",s=size,ax=axes,palette=color, alpha=alpha2)
   axes.loglog(alphc,betac[:,0],'-',color=color[0],label=r'$\langle k\rangle$=2',lw=lw)
   axes.loglog(alphc,betac[:,1],'-',color=color[1],label=r'$\langle k\rangle$=5',lw=lw)
   axes.loglog(alphc,betac[:,2],'-',color=color[2],label=r'$\langle k\rangle$=10',lw=lw)
   axes.loglog(alphc,betac[:,3],'-',color=color[3],label=r'$\langle k\rangle$=20',lw=lw)
   axes.loglog(alphc,betac[:,4],'-',color=color[4],label=r'$\langle k\rangle$=100',lw=lw)
   
   PlotAxes(axes,' ','','(b)')#Predicted  '+ r'$\alpha_c$,Theoretical  '+r'$\beta_c$   
   axes.legend(loc='lower left',bbox_to_anchor=(-0.01,-0.02),framealpha=0, ncol=2,fontsize=n_legend,handletextpad=0.1,markerscale=3.0,columnspacing=-0.1,borderaxespad=0.05)
   #axes.text(0.0000010,5,'(b)',fontdict = font_label)
   axes.set_yticks([0.00000001,0.000001,0.0001,0.01,1])

def plot_strategy(axes,result_path,color1, bgcolor):
    '''
    Plot the by varying the network structure but preserve the average degree 

    Parameters
    ----------
    axes : axes
        axes.
    color1 : str
        color.
    bgcolor : str
        background color.

    Returns
    -------
    None.

    '''
    
    #load the data
    inter_data = pd.read_csv(result_path + '/InterdisPhysics.csv')
    advogato_data = pd.read_csv(result_path + '/advogato.csv')
    delicious_data = pd.read_csv(result_path + '/soc-delicious.csv')
    fb_pages_artist = pd.read_csv(result_path + '/soc-fb-pages-artist.csv')
    
    #plot the scatter
    size = 350
    sns.scatterplot(data=inter_data, x='alphac_r', y= 'MC_betac',s=size,ax=axes, marker='o', color=color1[0])
    sns.scatterplot(data=advogato_data, x='alphac_r', y= 'MC_betac',s=size,ax=axes, marker='s', color=color1[2])
    sns.scatterplot(data=delicious_data, x='alphac_r', y= 'MC_betac',s=size,ax=axes, marker='X',color=color1[1])
    sns.scatterplot(data=fb_pages_artist, x='alphac_r', y= 'MC_betac',s=size,ax=axes, marker='P', color=color1[3])
    #sns.scatterplot(data=soc_twitter_Huawei, x='alphac_p', y= 'DMP_betac',s=size,ax=axes, marker='h', color=color1[4])
    
    #plot the theoretical lines 
    alphac = np.arange(0.05,3,0.1)
    ks = [2,5,10,20,100]
    lambd = 2.63
    [talphac,tbetac] = theoretical_results(alphac,ks,lambd)

    axes.loglog(talphac,tbetac[:,0],'-',color=color1[0],label=r'$\langle k\rangle$=2', lw=2)
    axes.loglog(talphac,tbetac[:,1],'-',color=color1[1],label=r'$\langle k\rangle$=5', lw=2)
    axes.loglog(talphac,tbetac[:,2],'-',color=color1[2],label=r'$\langle k\rangle$=10', lw=2)
    axes.loglog(talphac,tbetac[:,3],'-',color=color1[3],label=r'$\langle k\rangle$=20', lw=2)
    axes.loglog(talphac[0:3],tbetac[:,4][0:3],'-',color=color1[4],label=r'$\langle k\rangle$=100', lw=2)
   
    PlotAxes(axes,' ','','(c)')#Predicted '+ r'$\alpha_c$,Theoretical  '+r'$\beta_c$
    axes.set_xscale('log')
    axes.set_yscale('log')
    
    #mark the selected datasets
    #ms = 18
    #alpha1 = 0.5
    #mew = 2
    #complete_data = pd.read_csv(result_path+'/innovation_datasets.csv')
    #axes.plot(complete_data.iloc[23]['P_alphaC'],complete_data.iloc[23]['DMP_betac'], mfc='white', mec=bgcolor, marker='o', ms=ms, alpha=alpha1, mew = mew)#color1[0]
    #axes.plot(complete_data.iloc[52]['P_alphaC'],complete_data.iloc[52]['DMP_betac'], mfc='white', mec=bgcolor, marker='s', ms=ms, alpha=alpha1, mew = mew)#color1[2]
    #axes.plot(complete_data.iloc[53]['P_alphaC'],complete_data.iloc[53]['DMP_betac'], mfc='white', mec=bgcolor, marker='X', ms=ms, alpha=alpha1, mew = mew)#color1[1]
    #axes.plot(complete_data.iloc[55]['P_alphaC'],complete_data.iloc[55]['DMP_betac'], mfc='white', mec=bgcolor, marker='P', ms=ms, alpha=alpha1, mew = mew)#color1[3]
    
    fonts = 24
    font_label2 = {'family': "Arial", 'size':fonts, 'rotation':-10, 'color':bgcolor}
    font_label3 = {'family': "Arial", 'size':fonts, 'rotation':-15, 'color':bgcolor}
    font_label4 = {'family': "Arial", 'size':fonts, 'rotation':-38, 'color':bgcolor}
    font_label5 = {'family': "Arial", 'size':fonts, 'rotation':-15, 'color':bgcolor}

    dataname = ['InterdisPhysics', 'Advogato','Delicious', 'Fb-pages-artist']
    axes.text(0.33,0.11,dataname[0],fontdict=font_label2)
    
    axes.annotate("", xy=(0.06,0.35), xytext=(0.28,0.29),arrowprops=dict(arrowstyle="->",mutation_aspect=2, shrinkA=5,color=color1[0], lw=2))
    axes.text(0.06,0.34,'Disassortative',color=color1[0],rotation=-8,size=20)
    
    axes.annotate("", xy=(3.0,0.025), xytext=(1.4,0.073),arrowprops=dict(arrowstyle="->",mutation_aspect=2, shrinkA=5,color=color1[0],lw=2))
    axes.text(1.4,0.031,'Assortative',color=color1[0],rotation=-27,size=20)
    axes.text(0.9,0.0028,dataname[2],fontdict=font_label4)

    axes.text(0.19,0.01771116,dataname[1],fontdict=font_label3)
    axes.text(0.05,0.002,dataname[3],fontdict=font_label5)

if __name__ == '__main__':
   
   root_path = 'F:/work/work4_dynamic' #need to change the current path if run the code
   result_path = root_path + '/InnovationDiffusion/sfigure6/result'    
   figure_path = root_path + '/InnovationDiffusion/sfigure6/figure'
   
   #load the data 
   complete_data = pd.read_csv(result_path+'/innovation_datasets.csv')
   
   #classify the networks into several group
   data = classfiy_density(complete_data)
   
   #set the parameter
   lambd = 2.63
   ks = [2,5,10,20,100]
   alphac = np.arange(0.0001,30.0001,0.01)
   
   #calculate the theoretical result on uncorrelation network 
   [alphac,betac] = theoretical_results(alphac,ks,lambd)
   
   color = ["#4063a3","#ed6954","#34b6c6","#f4ac12",'#79ad41']
   color1 = ['#4063a3','#34b6c6','#79ad41','#ddc000','#d7aca1','#ed6954','#925951']
   bg_color = '#3b3a3e'
   
   #plot the result
   fig,ax = plt.subplots(1,3,figsize = (25,8), constrained_layout=True)
   
   #plot the efficience of diffusion for different kinds of networks
   #network are embedded in space spanned by alphac and betac
   plot_alpbacBetac(ax[0],complete_data,alphac,betac,color1)
   
   #classifying neworks over different average degree 
   plot_alpbacBetac_loglog(ax[1],data,alphac,betac,color1[:-2])
   
   #varying the network structure but preserve the average degree 
   plot_strategy(ax[2],result_path,color1,bg_color)
   
   fig.text(0.47,0.01,'Numerical  '+ r'$\alpha_c$', fontdict={'family': "Arial", 'size':38})
   fig.text(0.00,0.32,'Numerical  '+r'$\beta_c$', fontdict={'family': "Arial", 'size':38}, rotation = 'vertical')

   plt.savefig(figure_path+'/Relation.png', dpi=300)
   plt.savefig(figure_path+'/Relation.eps')
   plt.savefig(figure_path+'/Relation.pdf')

   