# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:34:15 2022

@author: Leyang Xue
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_log_error
    
def cross_validation(x, y):
    
    lr = LinearRegression()
    loo = LeaveOneOut()
    scores = cross_val_score(lr, x, y, cv=loo, scoring='neg_mean_squared_log_error') #计算模型的评分情况
    #print('score：', scores)
    return scores.mean()
    
def PlotAxes(ax,xlabel,ylabel,title=''):
    '''

    Parameters
    ----------
    ax : axes
        axes.
    xlabel : str
        xlable.
    ylabel : str
        ylabel.
    title : str, optional
        set the title. The default is ''.

    Returns
    -------
    None.

    '''
    font_label = {'family': "Arial", 'size':40}
    n_legend = 36
    ax.set_xlabel(xlabel, fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict =font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    #ax.locator_params(tight='True')
    #ax.ticklabel_format(style='sci')
    
def root_mean_squared_error(y_true, y_pred):
    '''
    This function calculates RMSE

    Parameters
    ----------
    y_true : array
        list of real numbers, true values.
    y_pred : array
        list of real numbers, predicted values.

    Returns
    -------
    value :float
        root mean squared error.

    '''
    
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt-yp)**2
    
    return np.sqrt(error/len(y_true))    

def mean_absolute_error(y_true, y_pred):
    '''
    This function calculates mae 

    Parameters
    ----------
    y_true : array
        list of real numbers, true values.
    y_pred : array
        list of real numbers, predicted values .

    Returns
    -------
    value :float
        mean absolute error.
    '''

    # initialize error at 0 
    error = 0 
    # loop over all samples in the true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate absolute error  
        # and add to error 
        error += np.abs(yt - yp) 
    # return mean error 
    return error / len(y_true) 
 
def r2(y_true, y_pred): 
    '''
    This function calculates r-squared score 

    Parameters
    ----------
    y_true : array
        list of real numbers, true values.
    y_pred : array
        list of real numbers, predicted values.

    Returns
    -------
    value : array
        r2 score .
    '''
    # calculate the mean value of true values 
    mean_true_value = np.mean(y_true) 
    
    # initialize numerator with 0 
    numerator = 0 
    # initialize denominator with 0 
    denominator = 0 
     
    # loop over all true and predicted values 
    for yt, yp in zip(y_true, y_pred): 
        # update numerator 
        numerator += (yt - yp) ** 2 
        # update denominator 
        denominator += (yt - mean_true_value) ** 2 
    
    # calculate the ratio 
    ratio = numerator/denominator 
    
    return 1-ratio


def PlotAxesSingle(ax,xlabel,ylabel,title,mode=False):
    
    fontsize = 28
    font_label = {'family': "Arial", 'size':fontsize}
    
    n_legend = 16
    #font_label = {'family': "Arial", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = {'family': "Arial", 'size':24})
    #ax.text(index[0],index[1],title,fontdict = font_label)
    ax.tick_params(direction='in', which='both',length =4, width=1, labelsize=24)
    if mode == True:
        ax.legend(loc='upper left',framealpha=0, fontsize=n_legend)
    
def fitlinearModel1(axes,x,y):
    '''
    plot the linear fitted model 

    Parameters
    ----------
    axes : axes
        axes.
    x : array
        variable x.
    y : array
        variable y.

    Returns
    -------
    None.

    '''
    color = '#4063a3'
    rx = x[:,0]
    ry = y[:,0] 
    model_parameter = np.polyfit(rx,ry,1)
    model_x =np.arange(min(rx), max(rx),0.01)
    model_y = np.polyval(model_parameter, model_x)

    axes.plot(rx, ry,'o',mec=color,mfc='white',mew=2, label='real networks')
    axes.plot(model_x, model_y,color='black', label=r'$\alpha_c=$'+str(round(model_parameter[1],2))+' + '+ str(round(model_parameter[0],2))+'LS')
    PlotAxesSingle(axes,'LS',r'$\alpha_c$','(a)', mode=True)
            
    subax = axes.inset_axes((0.60,0.1,0.3,0.3))
    subax.loglog(rx,ry,'o',mec=color,mfc='white',mew=2)
    subax.plot(model_x, model_y, '-',color='black')
    subax.set_xlabel('LS')
    subax.set_ylabel(r'$\alpha_c$')

def fitPowerModel(axes,rx,ry):
    '''
    plot the fitted power model 

    Parameters
    ----------
    axes : axes
        axes.
    rx : array
        variable x.
    ry : array
        variable y.

    Returns
    -------
    None.

    '''
    color = '#4063a3'

    x_log = np.log(rx)
    y_log = np.log(ry)
    [rbeta0,rbeta1,rybar] = LR(x_log,y_log,rx)
    
    model_x = np.arange(min(rx),max(rx),0.01)
    mode_y = np.exp(rbeta0)*np.power(model_x,rbeta1)
    axes.plot(rx, ry,'o',mec=color,mfc='white',mew=2, label='real network')
    axes.plot(model_x, mode_y, '-',color='black', label = r'$\alpha_c=%sLS^{%s}$'%(str(round(np.exp(rbeta0),2)),str(round(rbeta1,2))))
    PlotAxesSingle(axes,'LS',r'$\alpha_c$','(b)', mode=True)
    
    subax = axes.inset_axes((0.60,0.1,0.3,0.3))
    subax.loglog(rx,ry,'o',mec=color,mfc='white',mew=2)
    subax.plot(model_x, mode_y, '-',color='black')
    subax.set_xlabel('LS')
    subax.set_ylabel(r'$\alpha_c$')

def fitPowerModel_NM(axes,rx,ry):
    '''
    plot the fitted power model on model network

    Parameters
    ----------
    axes : axes
        axes.
    rx : array
        variable x.
    ry : array
        variable y.

    Returns
    -------
    None.

    '''
    color = '#4063a3'

    model_data= pd.read_csv(result_path+'/model_alphac.csv')
    mx = model_data['LS'].values.reshape(-1,1)
    my = model_data['alphac'].values.reshape(-1,1)
    mx_log = np.log(mx)
    my_log = np.log(my)
    [rbeta0,rbeta1,rybar] = LR(mx_log,my_log,rx)

    model_x = np.arange(min(rx),max(rx),0.01)
    mode_y = np.exp(rbeta0)*np.power(model_x,rbeta1)
    axes.plot(rx, ry,'o',mec=color,mfc='white',mew=2, label='real network')
    axes.plot(model_x, mode_y, '-',color='black', label = r'$\alpha_c=%sLS^{%s}$'%(str(round(np.exp(rbeta0),2)),str(math.ceil(rbeta1))))
    PlotAxesSingle(axes,'LS',r'$\alpha_c$','(c)', mode=True)
    
    subax = axes.inset_axes((0.60,0.1,0.3,0.3))
    subax.loglog(rx,ry,'o',mec=color,mfc='white',mew=2)
    subax.plot(model_x, mode_y, '-',color='black')
    subax.set_xlabel('LS')
    subax.set_ylabel(r'$\alpha_c$')

def LR(xlog,ylog,x):
    '''
    This function can estimate the parameter for the power model
    lny = beta0 lnx + beta1
    Parameters
    ----------
    xlog : float
        take a log for variable x.
    ylog : float
        take a log for variable y.
    x : float
        variable x.

    Returns
    -------
    beta0 : float
        estimated coefficient beta0.
    beta1 : float
        estimated coefficient beta1.
    predict : float
        predicted y using the fit model.
    '''
    
    lr = LinearRegression()
    lr.fit(xlog,ylog)
    beta1 = lr.coef_[0][0]
    beta0 = lr.intercept_[0]
    predict = np.exp(beta0)*np.power(x,beta1)#lmb0,lmb1

    return beta0,beta1,predict

def Leave_One_out(x_log, y_log):
    '''
    This function can estimate the accuracy of fitted model with the leave_one_out way 

    Parameters
    ----------
    x_log : float
        take a log for variable x.
    y_log : float
        take a log for variable y.

    Returns
    -------
    MSLE : float
        MSLE accuracy.

    '''
    loo = LeaveOneOut()
    data = np.stack((x_log[:,0],y_log[:,0]),axis=1)
    MSLE = []
    for train,test in loo.split(data):
        x_train = data[train][:,0]
        y_train = data[train][:,1]
        x_test = data[test][:,0]
        y_test = data[test][:,1]
        [beta0, beta1, predict]= LR(x_train.reshape(-1,1),y_train.reshape(-1,1),np.exp(x_test))
        MSLE.append(mean_squared_log_error(np.exp(y_test),predict))
            
    return MSLE

def Leave_One_out_model(x, y, mbeta0, mbeta1):
    '''
    The function is to estimate the MSLE by using the fitted model (based on the datasets of model network)
    to predict the tricritical point on real networks
    In fact,  the MSLE obtained by the leave one out way is equal to the MSLE obtained by predicting all real networks
    beacuse we here using the fitted model from dataset of model network
    
    Parameters
    ----------
    x : float
        variavle x .
    y : float
        variavle y .
    mbeta0 : float
        estimated coefficient from dataset of model network.
    mbeta1 : float
        estimated coefficient from dataset of model network.

    Returns
    -------
    MSLE : float
    accuracy of model
    '''
    loo = LeaveOneOut()
    dataset = np.stack((x[:,0],y[:,0]),axis=1)
    MSLE = []
    for train,test in loo.split(dataset):
        #x_train = dataset[train][:,0]
        #y_train = dataset[train][:,1]
        x_test = dataset[test][:,0]
        y_test = dataset[test][:,1]
        #[beta0, beta1, predict]= LR(x_train.reshape(-1,1),y_train.reshape(-1,1),np.exp(x_test))
        ym_predict = np.exp(mbeta0)*np.power(x_test,mbeta1)

        MSLE.append(mean_squared_log_error(y_test,ym_predict))
        
    return MSLE

def PlotAxesB(ax,xlabel,ylabel,title=''):
    '''
    This function is used to decorate the axes of histogram 

    Parameters
    ----------
    ax : axes
        axes.
    xlabel : str
        xlabel.
    ylabel : str
        ylabel.
    title : str, optional
        title. The default is ''.

    Returns
    -------
    None.

    '''
    font_label = {'family': "Arial", 'size':30}
    n_legend = 24
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',fontdict = font_label)
    ax.tick_params(direction='out', which='major',length =4, width=1, pad=1,labelsize=n_legend)
    #ax.locator_params(tight='True')
    #ax.ticklabel_format(style='sci')
    
def PlotLOOMSLE(x,y,figure_path):
    '''
    This function compare the accuracy of fitted model that are respectively obtained from real-world dataset and model datatsets 

    Parameters
    ----------
    x : array
        variable x from real-world datasets.
    y : array
        varibale y from real-world datasets.
    figure_path : str
        path to save the figure.

    Returns
    -------
    None.

    '''
    #1. real-world networks
    #directly predicting all train datasets: estimate the accuracy of powerlaw model on real network using the MSLE metrics
    x_log = np.log(x)
    y_log = np.log(y)
    [rbeta0,rbeta1,rybar] = LR(x_log,y_log,x)
    rMSLE= mean_squared_log_error(y,rybar)
    
    #leave one out way: estimate the accuracy of powerlaw model on real network using the MSLE metrics
    MSLE_score = Leave_One_out(x_log, y_log)
    MSLE_score_avg = np.average(MSLE_score)
    
    #2. model networks
    #load the datasets from model network
    model_data= pd.read_csv(result_path+'/model_alphac.csv')
    mx = model_data['LS'].values.reshape(-1,1)
    my = model_data['alphac'].values.reshape(-1,1)
    
    #directly predicting all train datasets:estimate the accuracy of powerlaw model on model network using the MSLE metrics
    mx_log = np.log(mx)
    my_log = np.log(my)
    [mbeta0,mbeta1,mybar] = LR(mx_log,my_log,mx)
    mMSLE= mean_squared_log_error(my,mybar)
    
    #leave one out: estimate the accuracy of powerlaw model on model network using the MSLE metrics
    mMSLE_score = Leave_One_out(mx_log, my_log)
    mMSLE_score_avg = np.average(mMSLE_score)
    
    #3. to predict the real networks by using the dataset of model network 
    mrMSLE_score= Leave_One_out_model(x,y,mbeta0,mbeta1)
    mrMSLE_score_avg = np.average(mrMSLE_score)
    
    #4: plot the figure to compare the result
    #here, we show the accuracy of fitted model obtained respectively from the dataset of real network and model network
    fontlabel = {'family': "Arial", 'size':24}
    color = '#4063a3'
    
    fig, ax = plt.subplots(1,2,figsize=(12,6), constrained_layout=True, sharey=True)
    ax[0].hist(MSLE_score,ec='white',color=color)
    ax[1].hist(mrMSLE_score,ec='white',color=color)
    ax[0].text(0.4,50, 'MSLE='+str(round(MSLE_score_avg,3)), fontdict =fontlabel)
    ax[1].text(0.5,50, 'MSLE='+str(round(mrMSLE_score_avg,3)), fontdict = fontlabel)
    ax[0].text(0.4,40, r'$\alpha_c=%sLS^{%s}$'%(str(round(np.exp(rbeta0),2)),str(round(rbeta1,2))),fontdict = fontlabel)
    ax[1].text(0.5,40, r'$\alpha_c=%sLS^{%s}$'%(str(round(np.exp(mbeta0),2)),str(math.ceil(mbeta1))),fontdict = fontlabel)
    ax[0].text(0.4,30, 'real-world networks', fontdict =fontlabel)
    ax[1].text(0.5,30, 'generated networks', fontdict = fontlabel)

    PlotAxesB(ax[0],'MSLE','Number of network','(a) ')
    PlotAxesB(ax[1],'MSLE','','(b)')
    plt.savefig(figure_path+'/model-comparision.png', dpi=300)
    plt.savefig(figure_path+'/model-comparision.eps')
    plt.savefig(figure_path+'/model-comparision.pdf')

def fit_model(data,figure_path):
    '''
    This function to select the model 

    Parameters
    ----------
    data : array
        datasets.
    figure_path : str
        path to save the figure.

    Returns
    -------
    None.

    '''
    x= data['LS'].values.reshape(-1,1)
    y = data['R_alphaC'].values.reshape(-1,1)
    
    #1. compare the accuracy between real-world and model network by using powermodel
    PlotLOOMSLE(x,y,figure_path)
    
    #2. compare the accuracy between different models
    fontlabel = {'family': "Arial", 'size':24}
    fig, ax = plt.subplots(1,3,figsize=(18,6), constrained_layout=True, sharey=True)
    
    #linear model on real network 
    fitlinearModel1(ax[0],x,y) 
    # power model on real network 
    fitPowerModel(ax[1],x,y)
    # power model on network model 
    fitPowerModel_NM(ax[2],x,y)
    
    plt.savefig(figure_path+'/model_selected.png',dpi=300)
    plt.savefig(figure_path+'/model_selected.eps')

    #existing results
    #y_predict =  data['P_alphaC']        
    #y_predict1 = 2.53*np.power(x,1)
    #mean_squared_log_error(y,y_predict)
    

if __name__ == '__main__':
    
    #set the path 
    root_path = 'F:/work/work4_dynamic' #need to change the current path if run the code
    result_path = root_path + '/InnovationDiffusion/figure3/result'
    figure_path = root_path + '/InnovationDiffusion/figure3/figure'
    
    #load the data 
    data = pd.read_csv(result_path+'/complete_datasets.csv')
    
    #calculate the score: MAE, R2, MSLE
    MAE_betac = mean_absolute_error(data['MC_betac'], data['DMP_betac'])
    MAE_alphac = mean_absolute_error(data['R_alphaC'], data['P_alphaC'])
    r2_betac = r2(data['MC_betac'], data['DMP_betac'])
    r2_alphac = r2(data['R_alphaC'], data['P_alphaC'])
    MSLE_betac = mean_squared_log_error(data['MC_betac'],data['DMP_betac'])
    MSLE_alphac = mean_squared_log_error(data['R_alphaC'], data['P_alphaC'])
    
    #generate the betas
    beta_min = 0.001
    beta_max = 1
    betas = np.arange(beta_min, beta_max, 0.01)
    
    #generate the alpha
    alpha_min= 0.001
    alpha_max = 101
    alphas = np.arange(alpha_min, alpha_max, 0.01)
    
    #set the color and fontlabel
    bg_color = '#3b3a3e'#CFD2CF
    palettte = ['#d7aca1','#4063a3','#ed6954','#34b6c6',"#f6c200",'#d7aca1','#5f9ed1' ,'#79ad41',"#40708c",'#dca215']
    font_label = {'family': "Arial", 'size':45}
    
    #plot the figure
    fig,ax = plt.subplots(1,2,figsize=(24,9),constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1,
        hspace=0., wspace=0.)
    
    ax[0].plot(betas, betas,ls='solid', color=bg_color, lw=1)
    ax[1].plot(alphas, alphas, ls='solid', color=bg_color,lw=1)
    sns.scatterplot(x='DMP_betac', y='MC_betac', hue='Type', style='Type', palette=palettte, data = data, ax=ax[0],s=400,legend=False)
    sns.scatterplot(x='R_alphaC', y='P_alphaC', hue='Type', style='Type', palette=palettte, data = data, ax=ax[1],s=400)
    #ax[0].loglog(data['DMP_betac'],data['MC_betac'],'o', color = colors[0], ms = msize, mec='white', mew=1.5)
    #ax[1].loglog(data['R_alphaC'],data['P_alphaC'],'o', color = colors[1], ms = msize,mec='white',mew=1.5)
    
    PlotAxes(ax[0],'Theoretical   '+ r'$\beta_c$','Numerical   '+r'$\beta_c$',title='')
    PlotAxes(ax[1],'Predicted   ' + r'$\alpha_c$','Numerical   ' + r'$\alpha_c$',title='')
    ax[1].legend(loc='center left',bbox_to_anchor=(0.95,0.5),framealpha=0, fontsize=32,handletextpad=0.3,markerscale=3)#

    ax[0].set_xlim(beta_min,beta_max)
    ax[0].set_ylim(beta_min,beta_max)
    ax[1].set_xlim(alpha_min,alpha_max)
    ax[1].set_ylim(alpha_min,alpha_max)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[1].set_xticks([0.01,1,100])    
    ax[1].set_yticks([0.01,1,100])

    ax[0].text(0.04,0.0019,'MSLE='+str(round(MSLE_betac,3)), fontdict = {'family': "Arial", 'size':38})
    ax[1].text(0.55,0.003,'MSLE='+str(round(MSLE_alphac,3)), fontdict = {'family': "Arial", 'size':38})
    
    ax[0].text(0.00015,1.2,'(a)', fontdict = font_label)
    ax[1].text(0.00005,130,'(b)', fontdict = font_label)

    plt.savefig(figure_path+'/figure3.eps')
    plt.savefig(figure_path+'/figure3.pdf')
    plt.savefig(figure_path+'/figure3.png',dpi=300)
    