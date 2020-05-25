"""
Plot helpers

"""

#https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import seaborn as sns

def plot_model_comparison(f1_scores,labels):
    len_labels = len(labels)
    colors = cm.gist_rainbow(np.linspace(0.,1.,len_labels))
    width = 0.6
    num_features = np.arange(len_labels)
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(num_features, f1_scores, width)
    
    ax.set_title('F1 score comparison across models',fontsize=14)
    ax.set_xlabel('Model',fontsize=13)
    ax.set_ylabel('Average F1 Score',fontsize=13)
    ax.set_xticks(num_features)
    ax.set_xticklabels(labels,rotation=0,va='top',ha='center',fontsize=12)
    ax.set_ylim([0,1])
    ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=6))
    
    for i,bar in enumerate(bars):
        bar.set_color(colors[i])
        height = bar.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=12)
    return fig, ax

def plot_training(log):
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(log.history['loss'],label='train')
    ax[0].plot(log.history['val_loss'],label='valid')
    ax[0].set_xlabel('Epoch',fontsize=12)
    ax[0].set_ylabel('Loss',fontsize=12)
    ax[0].set_title('Model loss',fontsize=14)
    ax[0].legend(loc='upper right',fontsize=12)
    ax[1].plot(log.history['accuracy'],label='train')
    ax[1].plot(log.history['val_accuracy'],label='valid')
    ax[1].set_xlabel('Epoch',fontsize=12)
    ax[1].set_ylabel('Accuracy',fontsize=12)
    ax[1].set_title('Model accuracy',fontsize=14)
    ax[1].legend(loc='lower right',fontsize=12)
    plt.show()
    return fig, ax

def legend_idx(dropout,dropout_space):
    leg_idx = []
    for i in range(len(dropout_space)):
        idx = np.where(dropout==dropout_space[i])[0][0]
        leg_idx.append(idx)
    return leg_idx

def customise_scatter(ax,xlabel=None,ylabel=None,zlabel=None):
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_zlabel(zlabel,fontsize=12) #,labelpad=6
    ax.xaxis.set_tick_params(labelsize=11,width=1.25)
    ax.yaxis.set_tick_params(labelsize=11,width=1.25)
    ax.zaxis.set_tick_params(labelsize=11,width=1.25)
    #ax.ticklabel_format(axis='z',style='sci',scilimits=(-4,-3))
    #ax.zaxis.major.formatter._useMathText = True
    for axis in ['bottom','left','top','right']:
        ax.spines[axis].set_linewidth(1.25)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.7)
    ax.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.7)
    ax.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.7)
    # make the axis patch transparent
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.dist = 13
    
def plot_legend(ax,leg_idx):
    leg = ax.legend(bbox_to_anchor=[1.0, 0.9], loc='lower center', fontsize=11, title="Dropout Rate",ncol=len(leg_idx), 
                    handletextpad=0.5, handlelength=1.0, columnspacing=1.0, borderaxespad=0, frameon=False) 
    #leg = ax.legend(bbox_to_anchor=[1.2, 0.85], loc='upper center', fontsize=11, title="Dropout", handletextpad=0.5, 
    #                frameon=False) 
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1) 
    leg.get_title().set_fontsize(12)
    #leg._legend_box.align='left'
    for marker in leg.legendHandles:
        marker.set_color('gray')    

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='cool',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    if xyplotlabels:
        plt.ylabel('True label',fontsize=12)
        plt.xlabel('Predicted label' + stats_text,fontsize=12)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title,fontsize=14)
