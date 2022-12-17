import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd



class ItemRegression:
    def __init__(self, predicts,r2Score=0, mseScore=0, label="", title="", color="b"):
        self.predicts = predicts
        self.color = color
        self.label = label
        self.title = title
        self.r2Score = r2Score
        self.mseScore = mseScore


class ItemClassification:
    def __init__(self,predicts,accuracy,precision,recall,f1Score,label,color="b"):
        self.predicts = predicts
        self.f1Score = f1Score
        self.accuracy=accuracy
        self.precision=precision
        self.recall=recall
        self.color=color
        self.label=label
 

   
def createTempDataFrame(items,columns=[],index=[]):
    return pd.DataFrame(list(map(lambda x:str(x),items)),columns=columns,index=index)


def plotConfusionMatrix(items,y_true,nrows=3,ncols=2,figSize=(11,12),textSize=20):
    
    plt.figure(figsize=figSize)
    f1SortedItems = sorted(items,key=lambda x:x.f1Score,reverse=True)
    
    for i,item in enumerate(f1SortedItems):
        ax=plt.subplot(nrows,ncols,i+1)
        disp=ConfusionMatrixDisplay.from_predictions(y_true,item.predicts,cmap=plt.cm.Blues,ax=ax)
        for labels in disp.text_.ravel():
            labels.set_fontsize(textSize)
        plt.title(f"{item.label} Model",fontdict={"size":textSize})

    plt.tight_layout()

    
def plotClassificationScoresHBars(items,nrows=4,ncols=1,figsize=(8,10),fontsize=None):
    plt.figure(figsize=figsize)
    
    f1SortedItems = sorted(items,key=lambda x:x.f1Score,reverse=False)
    plt.subplot(nrows,ncols,1)
    plotScoreBarh(list(map(lambda x:x.label,f1SortedItems)),
                  list(map(lambda x: x.f1Score,f1SortedItems)), title="F1",fontsize=fontsize)
    plt.xlim(0,1.1)

    accuracySortedItems = sorted(items,key=lambda x:x.accuracy,reverse=False)
    plt.subplot(nrows,ncols,2)
    plotScoreBarh(list(map(lambda x:x.label,accuracySortedItems)),
                  list(map(lambda x: x.accuracy,accuracySortedItems)), title="Accuracy",fontsize=fontsize)
    plt.xlim(0,1.1)

    precisionSortedItems = sorted(items,key=lambda x:x.precision,reverse=False)
    plt.subplot(nrows,ncols,3)
    plotScoreBarh(list(map(lambda x:x.label,precisionSortedItems)),
                  list(map(lambda x: x.precision,precisionSortedItems)), title="Precision",fontsize=fontsize)
    plt.xlim(0,1.1)


    recallSortedItems = sorted(items,key=lambda x:x.recall,reverse=False)
    plt.subplot(nrows,ncols,4)
    plotScoreBarh(list(map(lambda x:x.label,recallSortedItems)),
                  list(map(lambda x: x.recall,recallSortedItems)), title="Recall",fontsize=fontsize)
    plt.xlim(0,1.1)
    plt.tight_layout()
    
    
def plotClassificationScoreBars(items,nrows=2,ncols=3,figsize=(12,7),fontsize=None):
    
    plt.figure(figsize=figsize)
    f1SortedItems = sorted(items,key=lambda x:x.f1Score,reverse=True)

    for i,item in enumerate(f1SortedItems):
        plt.subplot(nrows,ncols,i+1)
        heights = [item.f1Score,item.accuracy,item.precision,item.recall]
        b=plt.bar(["f1","accuracy","precision","recall"],heights,label=item.label)
        plt.bar_label(b,padding=3,fontsize=fontsize)
        plt.title(item.label,fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.ylim(top=1.1)
        plt.margins(y=0.1) 
        plt.legend(loc="upper left")
 
    plt.tight_layout()
    
    
def plotHeatMap(corr, cmap="Reds", annot=True, fmt=".5f", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap=cmap, annot=annot, fmt=fmt)


def plotRegressionPredict(predicts, y_true, predictLabel="Predicted Label", valueLabel="True Label", xlabel="Date", ylabel="Close",
                          title="Test dataset predicted result"):
    plt.plot(predicts, label=predictLabel, c="green", alpha=0.7, linewidth=2.5)
    plt.plot(y_true, label=valueLabel, c="blue", alpha=0.7, linewidth=2.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


def plotScoreBarh(y, width, title="", xlabel="scores", ylabel="models",fontsize=None):

    b = plt.barh(y=y, width=width)
    plt.bar_label(b, padding=5,fontsize=fontsize)
    plt.margins(x=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(fontsize=fontsize)
    plt.title(title)


def plotTrueAndPredicts(itemsPredict, y_test, nrows=4, ncols=2, figSize=(12, 16)):
    sortedR2ScoreItems = sorted(itemsPredict, key = lambda x:x.r2Score,reverse=True)
    plt.figure(figsize=figSize)
    for i, item in enumerate(sortedR2ScoreItems):
        plt.subplot(nrows, ncols, i+1)
        plotRegressionPredict(item.predicts, y_test, title=item.title)

    plt.tight_layout()


def plotPredicts(itemsPredict, X_test, y_test, s=9, nrows=4, ncols=2,
                 xlabel="x", ylabel="y", isFirstRowOriginal=True, figSize=(12, 15),linewidth=4):
    plt.figure(figsize=figSize)
    
    itemsPredict = sorted(itemsPredict, key = lambda x:x.r2Score,reverse=True)
    
    if (isFirstRowOriginal):
        itemsPredict = list(np.array(itemsPredict))
        itemsPredict.insert(0, ItemRegression(y_test, label="Original"))

    if (isinstance(X_test, np.ndarray) and X_test.ndim > 1):
        X_test = tf.reduce_mean(X_test, axis=1)

    for i, item in enumerate(itemsPredict):
        plt.subplot(nrows, ncols, i+1)
        plt.scatter(X_test, y_test, s=s, label="real values", c="orange")
        if isFirstRowOriginal and i != 0:
            plt.plot(X_test, item.predicts, label=item.label,
                     linewidth=linewidth, c="darkblue")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.title(item.label)


def plotRegressionScoreBars(items, nrows=1, ncols=2, figSize=(8, 8)):
    plt.figure(figsize=figSize)

    r2SortedItems = sorted(items, key=lambda x: x.r2Score)
    plt.subplot(ncols, nrows, 1)
    plotScoreBarh(list(map(lambda x: x.label, r2SortedItems)), list(
        map(lambda x: x.r2Score, r2SortedItems)), title="R2 Scores")
    
    mseSortedItems = sorted(items, key=lambda x: x.mseScore, reverse=True)
    plt.subplot(ncols, nrows, 2)
    plotScoreBarh(list(map(lambda x: x.label, mseSortedItems)), list(
        map(lambda x: x.mseScore, mseSortedItems)), title="Mean Square Errors")

    
    plt.tight_layout()

    

