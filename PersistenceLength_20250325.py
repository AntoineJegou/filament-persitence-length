#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:26:33 2024

Calculates the persistence length and end-to-end distance correlation of actin filaments.

Author: ajegou
Updates:
    - Jan 31st: Bug fix: Removed absolute value of the cosine.
    - Feb 3rd: Removed the few first/last points of the snakes.
    - Feb 3rd: Added analysis of persistence length with sub-sampled snake points (as In Isambert et al, 1993).
    - Feb 28th: Added end-to-end distance correlation calculation.
    - Mar 4th: Added a limit to nber of analyzed filaments 
    - Mar 25th: for one actin species collect data from several experiments and plot the 
        average cosTheta(S) from them, with standard deviation: figure preparation.
"""


import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.optimize


def expFunc(x,lp):
    """
    Exponential function with the characteristic persisntence length parameter
    """
    return (np.exp(-1*x/(2*lp)))

def getSnakesfromFile(snakes_file):
    """
    Open output file(s) of the TSOAX software that contains the coordinates (x,y, frameNber) of the snakes 
    and store these coordinates into a pandas dataframe.
    """
    # Read the CSV file and comments lines that are not useful (eg replace '$' with '#')
    with open(snakes_file, 'r') as file:
        lines = file.readlines()
        lines = [line.replace('$', '#') for line in lines]
        lines = [line.replace('[', '#') for line in lines]
        lines = [line.replace('Tracks', '#') for line in lines]
    # Create a temporary preprocessed file
    with open('temp_file.csv', 'w') as file:
        file.writelines(lines)
    # Create the dataframe that contains the coordinates of the points along all snakes
    snakes = pd.read_csv('temp_file.csv', skiprows=32, comment='#', header=None, on_bad_lines='skip', sep=' ', skipinitialspace=True)
    snakes.columns = ['snakeID', 'point', 'x', 'y', 'z', 'intensity']
    # Remove the temporary file
    os.remove('temp_file.csv')
    # returns
    return snakes

def CosineCorrelation(snakes, filIndex, nbFilGroupSize, ptSpacing=4, removePtsEnds=0):
    """
    Compute the cosine of the angle between points along the curvilinear length 
    of snakes, to return the distribution along all snakes
    """
    # get the nber of lines of snakes
    snake_ids = snakes['snakeID'].unique()
    
    #nbFilaments = min(nbFilaments, len(snake_ids))
    # only look at the  filaments in the range
    snake_ids = snake_ids[filIndex:filIndex+nbFilGroupSize]
    #nbLines, nbColumns = snakes.shape
    # counter to accumulate nb of snakes and total length of filaments analyzed
    totalLength = 0
    maxLength = 0
    maxNbPointsInSnake = 500
    # print('processing snakeID:', end='')
    # create empty list to store data
    cosThetaList = [[] for a in range(maxNbPointsInSnake)]
    # for each snake (whatever the frame and length) read the top line and store into snakeID, x, y
    for snake_id in snake_ids:
            # get the snake
            x_raw = np.array(snakes.loc[snakes['snakeID']==snake_id,'x'])
            y_raw = np.array(snakes.loc[snakes['snakeID']==snake_id,'y'])
            # remove ends of snake
            x = x_raw[removePtsEnds:-removePtsEnds]
            y = y_raw[removePtsEnds:-removePtsEnds]
            # subsample
            #x = x_raw[::ptSpacing]
            #y = y_raw[::ptSpacing]
            points = np.column_stack((x,y))
            
            # get the tangent vectors along the snake
            tangentVector = np.diff(points, axis=0)
            # subsample
            tangentVector = tangentVector[::ptSpacing]
            
            # loop through all the possible distance between two points along a snake
            for j in np.arange(len(tangentVector)):
                # retrieve the list of already measure cosine for this distance j
                # between points along snakes
                cosThetaTemp = cosThetaList[j]
                for k in np.arange(len(tangentVector)-j):
                    cosTheta = 1 - scipy.spatial.distance.cosine(tangentVector[k],tangentVector[j+k])
                    cosThetaTemp.append(cosTheta)
                cosThetaList[j] = cosThetaTemp
            totalLength += len(tangentVector)
            maxLength = max(maxLength, len(tangentVector))
    return (cosThetaList, totalLength, len(snake_ids), maxLength)

def computeLp(actinType, snakeFile, removePtsEnds, ptSpacing, pixelSize, nbFilaments):
    """
    Compute the persistence length based on the cosine correlation of snakes
    """
    # get the snakes from this file
    snakes = getSnakesfromFile(f'{snakeFile}')
    nameSnakes = Path(snakeFile).stem
    
    maxNbFil = len(snakes['snakeID'].unique())
    
    minNbFil = nbFilaments
    ptDistance = pixelSize*ptSpacing
    
    
    for filIndex in np.arange(start=0, stop=maxNbFil, step= nbFilGroupSize):
        # compute cosine correlation
        cosThetaList, totalLength, nbFilamentsFile, maxLength = CosineCorrelation(snakes, filIndex=filIndex, nbFilGroupSize=nbFilGroupSize, ptSpacing=ptSpacing, removePtsEnds=removePtsEnds)
        print(f'{nameSnakes}: {nbFilaments} fil.; max/mean fil. length: {maxLength*ptDistance:.2f} / {totalLength*ptDistance/nbFilamentsFile:.2f} µm.')
        minNbFil = min(minNbFil, nbFilamentsFile)
        
        # compute the average and standard deviations of cosine from snake from this file
        cosThetaAvg = []
        cosThetaStd = []
        x = np.arange(maxLength-2)
        for i in x:
            a = np.array(cosThetaList[i])
            cosThetaAvg.append(np.mean(a))
            cosThetaStd.append(np.std(a))
                
        # plot the distributions of cosThetas as a function of the distance between snake points
        if False:
            for i in x[1:]:
                plt.hist(np.array(cosThetaList[i]), bins=50)
                plt.xlim((-1,1))
                plt.ylabel("frequency", size=14)
                plt.xlabel("cos($\\theta $)", size=14)
                plt.title(f'cos($\\theta $) distrib. for delta_dist = {i*ptDistance:.2f}')
                plt.show()
            plt.plot(np.arange(len(cosThetaStd))*ptDistance, cosThetaStd)
            plt.ylabel("cos($\\theta$) Stdev.", size=14)
            plt.xlabel("delta_dist (µm)", size=14)
            plt.title('Dependancy of cos stdev. on delta_dist.')
            plt.show()
            
        # plot the impact of the max length of the tangent vector analyzed on Lp.
        if True:
            lp = []
            lp_std = []    
            for maxL in np.arange(2,maxLength-2):
                # popt, pcov = scipy.optimize.curve_fit(expFunc, x[1:maxL]*ptDistance,cosThetaAvg[1:maxL], 
                #                                       p0=(10), sigma=[x for x in cosThetaStd[1:maxL]], absolute_sigma=False)
                popt, pcov = scipy.optimize.curve_fit(expFunc, x[1:maxL]*ptDistance,cosThetaAvg[1:maxL], 
                                                      p0=(10))
                lp.append(popt[0])
                lp_std.append(pcov[0][0])
                if ((maxL-1)*ptDistance>6) and ((maxL-1)*ptDistance<6.4):
                    # plot every fit
                    y1 = np.array(cosThetaAvg[0:maxL])-np.array(cosThetaStd[0:maxL])
                    y2 = np.array(cosThetaAvg[0:maxL])+np.array(cosThetaStd[0:maxL])
                    
                    #plt.errorbar(x[0:maxL]*ptDistance, cosThetaAvg[0:maxL], yerr=cosThetaStd[0:maxL], errorevery=1, fmt='.')
                    plt.fill_between(x[0:maxL]*ptDistance, y1=y1, y2=y2, interpolate=True, alpha=.15, color='grey')
                    plt.scatter(x[0:maxL]*ptDistance, cosThetaAvg[0:maxL])
                    plt.plot(x*ptDistance, expFunc(x*ptDistance, popt[0]), 'red', label='fitted line', linewidth=0.75)
                    plt.xlim((0,x[maxL]*ptDistance+2))
                    #plt.yscale('log')
                    plt.ylim((0.6,1.05))
                    plt.yticks(np.linspace(0.6, 1, num=5))
                    plt.xlabel('curvilinear length (µm)', size=14)
                    plt.ylabel("cos($\\theta $)", size=14)
                    plt.title(f'over {(maxL-1)*ptDistance:.1f} µm-long fil. Lp= {popt[0]:.2f} µm.')
                    plt.savefig('./'+Path(snakeFile).stem+f'{(maxL-1)*ptDistance:.1f}um_cosThetaFit.svg')
                    plt.show()
                    dataFigure = (x[0:maxL]*ptDistance, cosThetaAvg[0:maxL])
            plt.errorbar(np.arange(2,maxLength-2)*ptDistance, lp, yerr=lp_std, fmt='.--', capsize=2, label=snakeFile)
            plt.xlabel('max curvilinear distance (µm)', size = 14)
            plt.ylabel('Persistence Length (µm)', size=14)
            plt.xlim((0,20))
            plt.ylim((0,30))
            plt.legend()
            plt.title(snakeFile + ' - spacing:' + str(ptSpacing) + 
                      ' - nbFil.: ' + str(minNbFil) + ' - removePtsEnds: ' + str(removePtsEnds))
            plt.savefig('./'+Path(snakeFile).stem+'_persistenceLength.svg')
    plt.show()
    return(lp, lp_std, dataFigure, maxLength, ptDistance, nbFilamentsFile)

    
def plotFinalFigure(specie, dataPerSpecie):
    snake_files = glob.glob(f'./*{specie}_Snakes.txt')
    dataFinal = []
    for removePtsEnds in [6,]:
        for ptSpacing in [4,]:
            for nbFilaments in [1500,]:
                minNbFil = nbFilaments
                for snakeFile in snake_files:
                    lp, lp_std, dataFigure, maxLength, ptDistance, nbFilamentsFile = computeLp('actin Phalloidin', snakeFile, removePtsEnds, ptSpacing, pixelSize, nbFilaments)                
                    minNbFil = min(minNbFil, nbFilamentsFile)
                    dataFinal.append(dataFigure)
    
    x = dataFinal[0][0]
    y = np.zeros((len(dataFinal),len(dataFinal[0][1])))
    for i in np.arange(len(dataFinal)):
        for j in np.arange(len(dataFinal[0][1])):
            y[i,j] = dataFinal[i][1][j]
        plt.scatter(x,y[i,:])
    plt.savefig(f'./cosThetaCurveFinal_wIndividuals_{specie}.pdf')
    plt.show()  
        
      
    cosThetaAvgFinal = np.average(y,axis=0)
    cosThetaStdFinal = np.std(y,axis=0)
    popt, pcov = scipy.optimize.curve_fit(expFunc, x,cosThetaAvgFinal, 
                                          p0=(10))
    
    xfit = np.linspace(x[0],x[-1])
    
    #plt.errorbar(x,cosThetaAvgFinal, yerr=cosThetaStdFinal)
    # plot every fit
    y1 = cosThetaAvgFinal - cosThetaStdFinal
    y2 = cosThetaAvgFinal + cosThetaStdFinal
    
    #plt.errorbar(x[0:maxL]*ptDistance, cosThetaAvg[0:maxL], yerr=cosThetaStd[0:maxL], errorevery=1, fmt='.')
    plt.fill_between(x, y1=y1, y2=y2, interpolate=True, alpha=.15, color='grey')
    plt.scatter(x,cosThetaAvgFinal)
    plt.plot(xfit, expFunc(xfit,popt[0]), 'black', lw=0.75)
    #plt.plot(x*ptDistance, expFunc(x*ptDistance, popt[0]), 'red', label='fitted line', linewidth=0.75)
    plt.xlim((0,6.5))
    #plt.yscale('log')
    plt.ylim((0.6,1.05))
    plt.yticks(np.linspace(0.6, 1, num=5))
    plt.xlabel('curvilinear length (µm)', size=14)
    plt.ylabel("<cos($\\theta $)>", size=14)
    #plt.title(f'over {(maxL-1)*ptDistance:.1f} µm-long fil. Lp= {popt[0]:.2f} µm.')
    plt.savefig(f'./cosThetaCurveFinal_{specie}.pdf')
    plt.show()
    
    dataPerSpecie.append([cosThetaAvgFinal, cosThetaStdFinal])
    return(dataPerSpecie, x)

def LpFinal(species, dataPerSpecie):
    for i in np.arange(len(species)):
        specie = species[i]
        cosThetaAvgFinal = dataPerSpecie[i][0]
        cosThetaStdFinal = dataPerSpecie[i][1]
        popt, pcov = scipy.optimize.curve_fit(expFunc, x[1:],cosThetaAvgFinal[1:], 
                                              p0=(10)) #,sigma=cosThetaStdFinal[1:], absolute_sigma=True)
        y1 = cosThetaAvgFinal - cosThetaStdFinal
        y2 = cosThetaAvgFinal + cosThetaStdFinal
        
        #plt.errorbar(x[0:maxL]*ptDistance, cosThetaAvg[0:maxL], yerr=cosThetaStd[0:maxL], errorevery=1, fmt='.')
        plt.fill_between(x, y1=y1, y2=y2, interpolate=True, alpha=.15, color='grey')
        plt.scatter(x,cosThetaAvgFinal, label=specie)
        plt.plot(xfit, expFunc(xfit,popt[0]), 'black', lw=0.75)
        #plt.plot(x*ptDistance, expFunc(x*ptDistance, popt[0]), 'red', label='fitted line', linewidth=0.75)
    plt.xlim((0,6.5))
    #plt.yscale('log')
    plt.ylim((0.55,1.02))
    plt.yticks(np.linspace(0.6, 1, num=5))
    plt.xlabel('curvilinear length (µm)', size=14)
    plt.ylabel("<cos($\\theta $)>", size=14)
    plt.legend()
    plt.savefig('./Lp_graph.svg')
    plt.savefig('./Lp_graph.pdf')
    plt.show()
    

########
# MAIN #
########

# experiment parameters
pixelSize = 0.13 # in µm/pixel
nbFilGroupSize = 1500

dataPerSpecie = []
species = ['OcA', 'SpA', 'SpAm', 'ScA']
for specie in species:
    # print(snake_files)
    dataPerSpecie, x = plotFinalFigure(specie, dataPerSpecie)


