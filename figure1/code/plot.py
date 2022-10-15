# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:36:37 2022

@author: Leyang Xue
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib.image as mpimg
import pandas as pd

def load_data(result_path,networkname,key_words,alpha,p):
    '''
    Load the simulation result 
    
    Parameters
    ----------
    result_path : str
        reesult path.
    networkname : str
        network name.
    key_words : str 
        loath the result from DMP or MC.
    alpha : float
        internsity of conformity.
    p : float
        set the threshod so that we average the result with efficient simulation.

    Returns
    -------
    None.

    '''        
    
    file_name = networkname+'_'+key_words+'_'+str(alpha)+'.csv'
    rhos = np.loadtxt(result_path+'/'+file_name, delimiter=',')
    if key_words=='dmp':
        rhos_avg = np.average(rhos,axis=0)
    else:
        rhos_avg = np.zeros(rhos.shape[1])
        for i in np.arange(rhos.shape[1]):
            rho = rhos[:,i]
            rhos_avg[i] = np.average(rho[rho>p])
               
    return np.nan_to_num(rhos_avg)

def PlotAxes(ax,xlabel,ylabel,title=''):
    '''
    This function can decorate the axes

    Parameters
    ----------
    ax : ax
        axes.
    xlabel : str
        name of xlabel.
    ylabel : str
        name of ylabel.
    title : str, optional
        name of title. The default is ''.

    Returns
    -------
    None.

    '''
    font_label = {'family': "Arial", 'size':18}
    n_legend = 18
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = {'family': "Arial", 'size':20,'weight':'demibold'})
    ax.tick_params(direction='out', which='both',length =4, width=1, labelsize=n_legend)

def PlotText(ax,strs,color,yloc):
    '''
    This function set the parameter of text 
    
    Parameters
    ----------
    ax : ax
        axes.
    strs : str
        text content.
    color : str
        color.
    yloc : float
        coordinate from y.

    Returns
    -------
    None.

    '''
    font_label_color = {'family': "Arial", 'size':18, 'color':color}
    ax.text(-0.15,yloc,strs,rotation='vertical',fontdict = font_label_color)
    
def plot_phasepoint(ax,bg_color,data_color,betac,alphac,yup,ydown,row):
    '''
    Plot the criical point and tricritical point

    Parameters
    ----------
    ax : ax
        axes.
    bg_color : str
        background color.
    data_color : str
        color.
    betac : float
        critical point.
    alphac : float
        tricritical point.
    yup : float
        y coordinate from up.
    ydown : float
         y coordinate from down.
    row : int
        position of subfigure.

    Returns
    -------
    None.

    '''
    ylim  = 0.06
    font_label = {'family': "Arial", 'size':16}
    ax.set_facecolor(bg_color)
    ax.fill_between([betac,ylim], y1=0, y2=1,color=data_color[0])
    #ax.fill_between([betac,0.05], y1=alphac, y2=1,color=data_color[-1])
    ax.set_xlim(0,ylim)
    ax.set_ylim(0,1)
    ax.set_xticks([0.01,0.05])
    ax.set_yticks([0.0,1.0])
    
    ax.vlines(x=betac, ymin=alphac, ymax=1, color='black', linestyles='dashed',lw=2.2)
    ax.vlines(x=betac, ymin=0, ymax=alphac, color='black', linestyles='solid',lw=2.2)
    if row == 0:
        #ax.annotate('Discontinuous',(betac,yup),(betac+0.010,yup-0.04),arrowprops=dict(arrowstyle="->", ec="k"),size=14)
        #ax.annotate('Continuous',(betac,ydown),(betac+0.010,ydown-0.04),arrowprops=dict(arrowstyle="->", ec="k"),size=14)
    
        ax.text(0.004,0.22,'Vanished',rotation='vertical', fontdict=font_label)
        ax.text(0.025, 0.55,'Prevalent', rotation='horizontal',fontdict=font_label)
    ax.text(betac+0.002,alphac-0.05,r'$(\beta_c, \alpha_c)$', fontdict=font_label)
    ax.plot(betac,alphac,'o', color='black', ms=8, mfc='white')

    PlotAxes(ax,'','')   
    
if __name__ == '__main__':
    
    result_path = 'F:/work/work4_dynamic/code-2-1/innovation_diffution/figure1/result'
    figure_path = 'F:/work/work4_dynamic/code-2-1/innovation_diffution/figure1/figure'
    
    #set the parameter 
    networkName = ['ca-CondMat', 'email-Enron-Large','soc-Advogato']
    betas = np.array(list(map(lambda x:round(x,3),np.arange(0,0.301,0.005))))
    alphas = np.array(list(map(lambda x:round(x,2),np.arange(0.0,0.81,0.2))))
    p = 0.01
    
    #create the dataframe to save the rhos
    condmat_df = pd.DataFrame(index = betas, columns=['dmp_alphac2','dmp_alphac4','dmp_alphac6','dmp_alphac8','dmp_alphac0','mc_alphac2','mc_alphac4','mc_alphac6','mc_alphac8','mc_alphac0'])
    enron_df = pd.DataFrame(index = betas, columns=['dmp_alphac2','dmp_alphac4','dmp_alphac6','dmp_alphac8','dmp_alphac0','mc_alphac2','mc_alphac4','mc_alphac6','mc_alphac8','mc_alphac0'])
    advogato_df = pd.DataFrame(index = betas, columns=['dmp_alphac2','dmp_alphac4','dmp_alphac6','dmp_alphac8','dmp_alphac0','mc_alphac2','mc_alphac4','mc_alphac6','mc_alphac8','mc_alphac0'])
    
    #load the rhos on different networks
    for alpha in alphas:
        mc_alpha = 'mc_alphac'+str(int(alpha*10))
        dmp_alpha = 'dmp_alphac'+str(int(alpha*10))
        condmat_df[mc_alpha] = load_data(result_path,networkName[0], 'mc', alpha,p)
        condmat_df[dmp_alpha] = load_data(result_path,networkName[0], 'dmp', alpha,p)
        enron_df[mc_alpha] = load_data(result_path,networkName[1], 'mc', alpha,p)
        enron_df[dmp_alpha] = load_data(result_path,networkName[1], 'dmp', alpha,p)
        advogato_df[mc_alpha] = load_data(result_path,networkName[2], 'mc', alpha,p)
        advogato_df[dmp_alpha] = load_data(result_path,networkName[2], 'dmp', alpha,p)
    
    #set the parameter to decorate this figure
    msize = 12
    font_label = {'family': "Arial", 'size':18}
    color_set = ['#72aeb6','#4692b0','#2f70a1']#'#134b73'
    #advo_color = ['#5A8FB4','#4D80A0','#40708C']#'#36BAD0', '#42ACC4','#4E9EB8'
    #enron_color = ['#EE8067','#EE755E','#ED6954']#'#FFC60F','#F7C118','#EEBB20'
    #condmat_color = ['#9CA79E','#95A095','#8E998C']#'#7CE092','#69D183','#31A354'
    bg_color = '#dbdcdc'#CFD2CF
    text_color = '#979a9a' 
    alphas = np.array(list(map(lambda x:round(x,2),np.arange(0.0,0.81,0.4))))
    
    #plot the subfig 2
    fig,ax = plt.subplots(3,3,figsize=(6,6),constrained_layout=True, sharey=True, sharex=True)
    for j,alpha in enumerate(alphas):
        mc_alpha = 'mc_alphac'+str(int(alpha*10))
        dmp_alpha = 'dmp_alphac'+str(int(alpha*10))
        if j == 2:
            ax[0,j].plot(advogato_df.index,advogato_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize,label='Simulation')        
            ax[0,j].plot(advogato_df.index,advogato_df[dmp_alpha], ls='-', lw=4, color=color_set[j],label = 'Theoretical')
            ax[1,j].plot(enron_df.index,enron_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize)        
            ax[1,j].plot(enron_df.index,enron_df[dmp_alpha], ls='-', lw=4, color=color_set[j])
            ax[2,j].plot(condmat_df.index,condmat_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize)
            ax[2,j].plot(condmat_df.index,condmat_df[dmp_alpha],ls='-', lw=4, color=color_set[j])
            ax[0,j].set_title(r'$\alpha=$'+str(alpha), fontdict = font_label)
        else:
            ax[0,j].plot(advogato_df.index,advogato_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize)        
            ax[0,j].plot(advogato_df.index,advogato_df[dmp_alpha], ls='-', lw=4, color=color_set[j])
            ax[1,j].plot(enron_df.index,enron_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize)        
            ax[1,j].plot(enron_df.index,enron_df[dmp_alpha], ls='-', lw=4, color=color_set[j])
            ax[2,j].plot(condmat_df.index,condmat_df[mc_alpha],'o',mfc='white', mec=color_set[j], mew=0.8, ms=msize)
            ax[2,j].plot(condmat_df.index,condmat_df[dmp_alpha],ls='-', lw=4, color=color_set[j])
            ax[0,j].set_title(r'$\alpha=$'+str(alpha), fontdict = font_label)

        if j !=0: 
            PlotAxes(ax[0,j],'','')      
            PlotAxes(ax[1,j],'','')
            
    PlotAxes(ax[0,0],'','')
    PlotText(ax[0,0],'CondMat',text_color,yloc=0.15)
    PlotAxes(ax[1,0],'','Proportion of recovered nodes, '+r'$R(\infty)/N$'+'\n\n')
    PlotText(ax[1,0],'Enron-Large',text_color,yloc=0.05)
    PlotAxes(ax[2,0],'','')
    PlotText(ax[2,0],'Advogato',text_color,yloc=0.15)
    PlotAxes(ax[2,1],'\nIntrinsic attractiveness, '+r'$\beta_0$','')
    PlotAxes(ax[2,2],'','')
    
    ax[0,0].text(-0.27,1.1,'(b)', fontdict = {'family': "Arial", 'size':24})
    ax[2,2].set_xticks([0.05,0.25])
    ax[0,2].set_facecolor(bg_color)
    ax[0,1].set_facecolor(bg_color)
    ax[1,2].set_facecolor(bg_color)
    ax[2,2].set_facecolor(bg_color)
    ax[0,2].legend(loc='lower left', bbox_to_anchor=(0.038, 0.1), framealpha=0, fontsize=11)
    
    plt.savefig(figure_path+'/figure1-fig2.eps')
    plt.savefig(figure_path+'/figure1-fig2.png',dpi=300)
    #plt.savefig(figure_path+'/figure1-fig2.pdf')
    
    #plot the subfig 3
    fig,ax = plt.subplots(3,1,figsize=(2.8,6), sharex=True, sharey=True, constrained_layout=True)
    plot_phasepoint(ax[0],bg_color,data_color=color_set,betac=0.0148,alphac=0.28,yup=0.92,ydown=0.10, row=0)
    plot_phasepoint(ax[1],bg_color,data_color=color_set,betac=0.0087,alphac=0.45,yup=0.92,ydown=0.10, row=1)
    plot_phasepoint(ax[2],bg_color,data_color=color_set,betac=0.0279,alphac=0.8,yup=0.92,ydown=0.10, row=2)
    PlotAxes(ax[2],'   \n','')
    PlotAxes(ax[1],'','Strength of conformity, '+r'$\alpha$'+'\n')
    #ax[0].text(0.02,1.1,' ',fontdict = {'family': "Arial", 'size':20,'weight':'demibold'})
    ax[0].text(-0.0265,1.1,'(c)', fontdict = {'family': "Arial", 'size':24})

    plt.savefig(figure_path+'/figure1-fig3.eps')
    plt.savefig(figure_path+'/figure1-fig3.png',dpi=300)