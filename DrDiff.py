#!/usr/bin/env python3
#coding: utf8
################################################################################
#                                                                              #
# Calculate the Diffusion Coefficient (D(Q)) and the Free-Energy (F_Stoch(Q))  #
#   using the Stochastic Approach                                              #
#                                                                              #
#                                                                              #
# Contribute by:                                                               #
#  Vinícius de Godoi Contessoto                                                #
#  Frederico Campos Freitas                                                    #
#  Ronaldo Junio de Oliveira                                                   #
#                                                                              #
#                                                                              #
# python StochasticDiffusion.py trajectory_file                                #
#                                                                              #
#  trajectory (just one column - coordinate)                                   #
#                                                                              #
#                                                                              #
# PS: Need to install some libraries: numpy, scipy and itertools               #
#                                                                              #
################################################################################

import sys
import os
from itertools import islice
import numpy as np
#import scipy as sc
#import progressbar
#from time import sleep
from scipy import stats
from scipy import integrate
from scipy.signal import argrelmin
from scipy.signal import savgol_filter



################################################################################
# Function to print Free Energy and the Histogram to data file                 #
#   F(Q) = - np.log [number-of-states(Q)]                                      #
################################################################################
def Free_energy_Histogram_Q(filename, Qf, nbins):
    #Histogram calculation
    #hist, bins=np.histogram(Qf[::1], int(np.ceil(np.max(Qf)/Qbins)), density=True) ##
    hist,bins = np.histogram(Qf[::1], nbins, density=True) ##
    #hist, bins=np.histogram(Qf[::1], 28, density=True)
    bins_center = (bins[:-1] + bins[1:])/2.0 ## Average of bins positions to plot
    #np.savetxt('hist_' + filename + '.dat', np.c_[bins_center, hist]) ## Write Histogram file
    FQ = -1*np.log(hist) ## Free Energy calculation
    FG = savgol_filter(FQ, 5, 3, mode='nearest')
    Free = np.c_[bins_center, FQ]
    id = argrelmin(FG)[-1][-1]
    Free[:,1] = Free[:,1]-Free[:,1][id]
    np.savetxt('Free_energy_' + filename + '.dat', Free) ## Write Free Energy file
    #print('Coordinate Histogram and Free Energy calculated')
    #print('################################################')
    return
################################################################################

################################################################################
# Function to print Histogram of t stes to data file                           #
#                                                                              #
################################################################################
def Jump_Histogram(filename, Qf):
    hist, bins = np.histogram(Qf[::1], density=True) ## Histogram calculation
    bins_center = (bins[:-1] + bins[1:])/2.0 ## Average of bins positions to plot
    #np.savetxt('H_' + filename + '.dat', np.c_[bins_center, hist]) ## Write Histogram file
    return
################################################################################


################################################################################
# Function to check files output and delete lines with "nan" and "inf"         #
#                                                                              #
################################################################################

def CheckFiles(q):
    aw = os.listdir('.')
    bw = set(q).intersection(aw)
    for w in bw: # Delete lines with "nan" and "inf" inside.
        ff = open(w, "r+")
        dd = ff.readlines()
        ff.seek(0)
        for z in dd:
            if (("nan" not in z) and ("inf" not in z) and ("-inf" not in z)):
                ff.write(z)
        ff.truncate()
        ff.close()
    print('################## CHECKED ########################')
    return

################################################################################

################################################################################
# Function to exclude invalid values for n-dimensional matrix                  #
#                                                                              #
################################################################################

def excludeinvalid(M):
    M = M[~np.isnan(M).any(axis=1)]
    M = M[~np.isinf(M).any(axis=1)]
    M = M[~np.isneginf(M).any(axis=1)]
    return M
################################################################################


################################################################################
# Function to exclude invalid values for one-dimensional matrix                #
#                                                                              #
################################################################################

def excludeinvalid1D(M):
    N = M[~np.isnan(M)]
    O = N[~np.isinf(N)]
    P = O[~np.isneginf(O)]
    return P
################################################################################


################################################################################
# Function to calculate ptpx                                                   #
#                                                                              #
################################################################################

def ptpx(beta, Qzero, Qone, DQ, G):
    ptpxM = np.empty(shape=[0,2])
    for x in DQ[:,0]:
        if x >= Qzero and x <=Qone:
            xphi, unphi = phi(beta, Qzero, Qone, DQ, G, x)
            ptpxM = np.append(ptpxM, [[x, 2*xphi*(1-xphi)]], axis=0)
    return ptpxM

################################################################################

################################################################################
# Function to calculate t_{folding} using Kramers equation                     #
#                                                                              #
################################################################################

def calctau(beta, Qinit, Qzero, Qone, DQ, G):
    atau = 0
    utau = 0
    DQ = np.asarray(DQ)
    G = np.asarray(G)
    G = excludeinvalid(G)
    DQ = excludeinvalid(DQ)
    tau = np.empty(shape=[0,3])
    taul = np.empty(shape=[0,3])
    idxzero = (np.abs(G[:,0]-Qzero)).argmin() #get index of Q value close to Qzero
    idxone = (np.abs(G[:,0]-Qone)).argmin() #get index of Q value close to Qone
    #np.seterr(divide='ignore', invalid='ignore')
    if Qzero < Qone: x=1
    else: x=-1
    for Qj in G[:,0][idxzero:idxone+x:x]: #summing from Qzero to Qone
        irow, icol = np.where(DQ == Qj)
        jrow, jcol = np.where(G == Qj)
        tau = np.empty(shape=[0,3])
        idxinit = (np.abs(G[:,0]-Qinit)).argmin()
        idxj = (np.abs(G[:,0]-Qj)).argmin()
        if Qinit < Qj: y=1
        else: y=-1
        for Qk in G[:,0][idxinit:idxj+y:y]: #summing from Qinit to Qj
            krow, kcol = np.where(G == Qk)
            if (np.size(irow) != 0 and np.size(jrow) != 0 and np.size(krow) != 0):
                if not ((abs(float(G[np.int(jrow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1])+abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(jrow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1])-abs(3*np.nanstd(G, axis=0)[1])))):
                    GQ1 = (float(G[np.int(jrow[0]), 1]))
                    err1 = (float(G[np.int(jrow[0]), 2]))
                else:
                    GQ1 = np.nan
                    err1 = np.nan
                if not ((abs(float(G[np.int(krow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1])+abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(krow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1])-abs(3*np.nanstd(G, axis=0)[1])))):
                    GQ2 = (float(G[np.int(krow[0]), 1]))
                    err2 = (float(G[np.int(krow[0]), 2]))
                else:
                    GQ2 = np.nan
                    err2 = np.nan
                utau = ((np.exp(beta*(GQ1-GQ2)))/(float(DQ[np.int(irow[0]), 1]))) #calculating t_folding/unfolding
                if float(DQ[np.int(irow[0]), 1]) !=0:
                    uncerutau = np.absolute(utau)*np.sqrt(np.square(beta)*(np.square(err1)+np.square(err2))+np.square(float(DQ[np.int(irow[0]), 2])/float(DQ[np.int(irow[0]), 1])))
                else:
                    uncerutau = 0
            else:
                utau = 0
                uncerutau = 0
            tau = np.append(tau, [[Qk, utau, uncerutau]], axis=0)
            tau = excludeinvalid(tau)
        inttau = integrate.cumtrapz(tau[:,1], tau[:,0], axis=0, initial=tau[0,1])[-1] #inner integral
        uncertau = inttau*np.sqrt(np.mean(np.square(excludeinvalid1D(tau[:,2]/tau[:,1])))) #estimating error in inner integral
        taul = np.append(taul, [[Qj, inttau, uncertau]], axis=0)
        taul = excludeinvalid(taul)
    inttaul = integrate.cumtrapz(taul[:,1], taul[:,0], axis=0, initial=taul[0,1])[-1] #outer integral
    uncerttaul = inttaul*np.sqrt(np.amax(np.square(excludeinvalid1D(taul[:,2]/taul[:,1])))) #estimating error in inner integral
    return inttaul, uncerttaul

################################################################################


################################################################################
# Function to calculate analytical mtpt                                        #
#                                                                              #
################################################################################
def calcmtpt(beta, Qzero, Qone, DQ, G):
    DQ = np.asarray(DQ)
    G = np.asarray(G)
    #left part of the integral
    vlint = simpleint(testcalc, lcoreint, beta, Qzero, Qone, G, DQ)
    intlintegral = integrate.cumtrapz(vlint[:,1], vlint[:,0], axis=0, initial=vlint[0,1])[-1] #left integral from Qunf to Qfold
    #right part of integral
    vrint = simpleint(testcalc, rcoreint, beta, Qzero, Qone, G, DQ)
    intrintegral = integrate.cumtrapz(vrint[:,1], vrint[:,0], axis=0, initial=vrint[0,1])[-1] #right integral from Qunf to Qfold
    inttpt = intlintegral*intrintegral
    np.seterr(divide='ignore', invalid='ignore')
    unmtpt = np.absolute(inttpt)*np.sqrt(np.mean(np.square(excludeinvalid1D(vlint[:,2]/vlint[:,1]))) + np.mean(np.square(excludeinvalid1D(vrint[:,2]/vrint[:,1])))) #use max uncertainty evaluated in both integral combinations
    return inttpt, unmtpt
################################################################################

################################################################################
# Equation for mtpt - right integral                                           #
#                                                                              #
################################################################################
def rcoreint(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):
    val = ((np.exp(beta*(GQ1)))/(float(DQ[np.int(irow[0]), 1]))) #calculating rintegral
    if float(DQ[np.int(irow[0]), 1]) !=0:
        uncert = np.absolute(val)*(np.sqrt(np.square(beta)*np.square(unc)+np.square(float(DQ[np.int(irow[0]), 2])/float(DQ[np.int(irow[0]), 1]))))
    else:
        uncert = 0
    return val, uncert


################################################################################
# Equation for mtpt - left integral                                            #
#                                                                              #
################################################################################
def lcoreint(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):
    xphi, unphi = phi(beta, Qzero, Qone, DQ, G, Qx)
    val = (np.exp(-1*beta*(GQ1))*xphi*(1-xphi)) #calculating lintegral
    if GQ1 != 0 and xphi !=0:
        uncert = np.absolute(val)*beta*np.absolute((-1*beta*(GQ1))*xphi*(1-xphi))*(np.sqrt(np.square(unc/GQ1)+np.square(unphi/xphi)))
    else:
        uncert = 0
    return val, uncert

################################################################################
# Function to \phi(x)                                                          #
#                                                                              #
################################################################################
def phi(beta, Qzero, Qone, DQ, G, qx):
    DQ = np.asarray(DQ)
    G = np.asarray(G)
    vlowphi = simpleint(testcalc, equationphi, beta, Qzero, Qone, G, DQ)
    intlowphi = integrate.cumtrapz(vlowphi[:,1], vlowphi[:,0], axis=0, initial=vlowphi[0,1])[-1] #denominator integral from Qzero to Qone
    vupphi =  simpleint(testcalc, equationphi, beta, Qzero, qx, G, DQ)
    intupphi = integrate.cumtrapz(vupphi[:,1], vupphi[:,0], axis=0, initial=vupphi[0,1])[-1] #numerator integral from Qzero to Q
    phix = (intupphi/intlowphi)
    uncphi = np.absolute(phix)*np.sqrt(np.mean(np.square(excludeinvalid1D(vupphi[:,2]/vupphi[:,1]))) + np.mean(np.square(excludeinvalid1D(vlowphi[:,2]/vlowphi[:,1])))) #use max uncertainty evaluated in both integral combinations
    return phix, uncphi
################################################################################

################################################################################
# Function to evaluate values to a simple integral                             #
#                                                                              #
################################################################################
def simpleint(calctest, funcion, beta, Qzero, Qone, G, DQ):
    G = excludeinvalid(G)
    DQ = excludeinvalid(DQ)
    sampledvalues = np.empty(shape=[0,3]) #initializing two column numpy array
    idxzero = (np.abs(G[:,0]-Qzero)).argmin() #get index of Q value close to Qzero
    idxone = (np.abs(G[:,0]-Qone)).argmin() #get index of Q value close to Qone
    if Qzero < Qone: x=1
    else: x=-1
    for Qx in G[:,0][idxzero:idxone+x:x]:  #summing from Qzero to Qone
        irow, icol = np.where(DQ == Qx)
        jrow, jcol = np.where(G == Qx)
        value, uncertainty = calctest(funcion, irow, jrow, G, DQ, beta, Qx, Qzero, Qone)
        sampledvalues = np.append(sampledvalues, [[Qx, value, uncertainty]], axis=0)
    sampledvalues = excludeinvalid(sampledvalues)
    return sampledvalues
################################################################################

################################################################################
# Function to calculate core of \phi(x) integral                               #
#                                                                              #
################################################################################
def equationphi(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):
    val = ((np.exp(beta*(GQ1)))/(float(DQ[np.int(irow[0]), 1]))) #calculating phi core
    if float(DQ[np.int(irow[0]), 1]) != 0:
        uncert = np.absolute(val)*(np.sqrt(np.square(beta)*np.square(unc)+np.square(float(DQ[np.int(irow[0]), 2])/float(DQ[np.int(irow[0]), 1]))))
    else:
        uncert = 0
    return val, uncert
################################################################################

################################################################################
# Test to avoid invalid values and evaluate the chosen equation                #
#                                                                              #
################################################################################
def testcalc(eq, irow, jrow, G, DQ, beta, Qx, Qzero, Qone):
    eval = 0
    if (np.size(irow) != 0 and np.size(jrow) != 0):
        if not ((abs(float(G[np.int(jrow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1])+abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(jrow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1])-abs(3*np.nanstd(G, axis=0)[1])))):
            GQ1 = (float(G[np.int(jrow[0]), 1]))
            unci = (float(G[np.int(jrow[0]), 2]))
        else:
            GQ1 = np.nan
            unci = np.nan
        eval, uncer = eq(irow, jrow, G, DQ, beta, GQ1, unci, Qx, Qzero, Qone)
    else:
        eval = 0
        uncer = 0
    return eval, uncer
################################################################################


################################################################################
# Function to calculate mfpt, mtpt and number of transitions from trajectory   #
#                                                                              #
################################################################################
def calcttrajectory(Qzero, Qone, Qtr):
    tAB = tBA = tTP = nAB = nBA = t0  = t1 = t2 = 0
    resultsAB = np.empty(shape=[0,4])
    resultsBA = np.empty(shape=[0,4])
    resultsTP0 = np.empty(shape=[0,4])
    resultsTP2 = np.empty(shape=[0,4])
    if Qtr[0] <= Qzero: s = 0 #Defines initial state. A is s==0
    elif Qtr[0] >= Qone: s = 2 #B is s==2
    else: s = 1 # transition state
    for i in range(np.size(Qtr)):
        if s == 0:
            if Qtr[i] <= Qzero: t1 = i+1 #identify last time when Q is lower than Q0
            if Qtr[i] >= Qone: #identify when Q is greater than Q1
                t2 = i+1
                nAB = nAB + 1 #count a transition
                s = 2
                resultsAB = np.append(resultsAB, [[nAB, t0, t1, (t2-t0)]], axis=0) #add results in a row
                resultsTP0 = np.append(resultsTP0, [[nAB, t1, t2, (t2-t1)]], axis=0)
                t0 = t2
                t1 = t2
        elif s == 2:
            if Qtr[i] >= Qone: t1 = i+1 #identify last time when Q is greater than Q1
            if Qtr[i] <= Qzero: #identify when Q is lower than Q0
                t2 = i+1
                nBA = nBA +1 #count a transition
                s = 0
                resultsBA = np.append(resultsBA, [[nBA, t0, t1, (t2-t0)]], axis=0)
                resultsTP2 = np.append(resultsTP2, [[nBA, t1, t2, (t2-t1)]], axis=0)
                t0 = t2
                t1 = t2
        elif s == 1:
            if Qtr[i+1] <= Qzero: s = 0
            elif Qtr[i+1] >= Qone: s = 2
    tAB = (np.sum(resultsAB, axis=0)[3])/nAB
    #print(tAB)
    #print(np.nanmean(resultsAB, axis=0)[3])
    stdtAB = np.nanstd(resultsAB, axis=0)[3]
    tBA = (np.sum(resultsBA, axis=0)[3])/nBA
    stdtBA = np.nanstd(resultsBA, axis=0)[3]
    nTPAB = (resultsTP0[(np.size(resultsTP0, axis=0)-1)][0])
    tTPAB = (np.sum(resultsTP0, axis=0)[3])/nTPAB
    stdtTPAB = np.nanstd(resultsTP0, axis=0)[3]
    nTPBA = (resultsTP2[(np.size(resultsTP2, axis=0)-1)][0])
    tTPBA = (np.sum(resultsTP2, axis=0)[3])/nTPBA
    stdtTPBA = np.nanstd(resultsTP2, axis=0)[3]
    nTP = (resultsTP0[(np.size(resultsTP0, axis=0)-1)][0])+(resultsTP2[(np.size(resultsTP2, axis=0)-1)][0])
    tTP = ((np.sum(resultsTP0, axis=0)[3])+(np.sum(resultsTP2, axis=0)[3]))/(nTP)
    #np.savetxt('Results.dat', resultsAB) #to print results  matrix_m in a file
    return tAB, tBA, nTPAB, tTPAB, nTPBA, tTPBA, nAB, nBA, nTP, tTP, stdtAB, stdtBA, stdtTPAB, stdtTPBA
################################################################################


################################################################################
## Global variables declaration

EQ = 10 # Equilibration Steps - Value to ignore the first X numbers from the traj file
Qbins = 1 # Estimated bin width used to analyze the trajectory
tmax = 6 # Default 6
tmin = 2 # Default 2
time_step = 0.0005 # time step value used to save the trajectory file - Default 0.001
Snapshot = 50 # Snapshots from simulation
CorrectionFactor = time_step*Snapshot
beta = 1 # beta is 1/k_B*T
Q_zero = 80 # transition boundaries
Q_one = 230


#print('################################################')
print('Equilibration Steps = ', EQ)
print('Bin width read from trajectory = ', Qbins)
print('Time Step = ', time_step)
print('Snapshot = ', Snapshot)
print('tmax = ', tmax, '| tmin = ', tmin)
print('beta = ', beta)
print('Transition state boundaries = ', Q_zero, ' and ', Q_one)
#print('################################################')

################################################################################

def main():

    if len(sys.argv) > 1: ## To open just if exist  file in argument
        arg = sys.argv[1]
    else:
        print('No trajectory file found - Please insert a trajectory file')
        sys.exit()
    try:
        f = open(arg, 'r')
    except (IOError) as errno:
        print('I/O error. %s' % errno)
        sys.exit()
    print('Reading trajectory file')
    #print('################################################')
    #Pbar = progressbar.ProgressBar(term_width=53, widgets=['Working: ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    Q = np.asarray([float(line.rstrip()) for line in islice(f, EQ, None)]) # Save the coordinate value skipping the Equilibration steps
    Qmax = np.max(Q) # take the max and min value
    Qmin = np.min(Q)
    print('From trajectory file')
    print('Qmax =', Qmax, '| Qmin =', Qmin)
    print('Mean(Q) =', np.mean(Q), '| Std(Q) =', np.std(Q))
    print('Std(Q)/Mean(Q) =', np.std(Q)/np.mean(Q))
    if ((Q_zero < Qmin) or (Q_one > Qmax)):
        Q_zero = (Qmin + 0.2*abs(Qmin))
        Q_one = (Qmax - 0.2*abs(Qmax))
        print('The transition state boundary was mischoosed. Your new transition state boundaries are = ', Q_zero, ' and ', Q_one)
    else:
        print('The analysis will start.')
    #print('################################################')
    nbins = np.int(np.ceil((Qmax-Qmin)/Qbins))
    Free_energy_Histogram_Q(arg, Q, nbins) ## Call function to Free Energy and Histogram
    DQ=[]
    VQ=[]

    #Add_end=np.arange(Qmax, Qmax+tmax+1) ## Vector just to avoid the empty end of file
    Add_end=np.linspace(Qmax, Qmax+Qbins, tmax+1)
    Q = np.concatenate((Q, Add_end))
    #Pbar.start()
    #for Qi in np.arange(Qmin+1.0, Qmax-1.0, Qbins): #just for test
    bins = np.linspace(Qmin, Qmax, nbins+1)
    binscenter = np.delete((bins[:-1] + bins[1:])/2.0, -1)
    for Qi in binscenter:
    #for Qi in np.delete(np.linspace(Qmin, Qmax, nbins+1), -1):
        Q_index = np.array(np.where( (Qi + Qbins > Q) & (Qi - Qbins < Q)))[0] ## Find the Value of Qi with a bin in trajectory
        x=Q[Q_index]
        D=[]
        V=[]
        #Pbar.update( (Qi-Qmin)*100/(Qmax-Qmin))
        #sleep(0.01)
        for t in range(tmin, tmax): # Loop of times for linear regression
            #print(Q_index+t, t, Q[Q_index])
            y=Q[Q_index+t]
            ####################  Print Historgram Do not delete it #######################################
            Jump_Histogram(str(Qi) + '_' +str(t), y) ## Write Histograms with each t for each coordinate Qc
            ###############################################################################################
            #print(Qi, t, np.var(y), np.mean(y))
            D.append([t, 0.5*np.var(y)]) # Variance calculation - sigma^2
            V.append([t, np.mean(y)]) # Mean of histogram calculation - Qc

        sloped, interceptd, r_valued, p_valued, std_errd = stats.linregress(D) # Linear regression - sloped is the difusion
        slopev, interceptv, r_valuev, p_valuev, std_errv = stats.linregress(V) # Linear regression - slopev is the drift
        DQ.append([Qi, sloped/CorrectionFactor, std_errd]) # Save Diffusion for each coordinate value
        VQ.append([Qi, slopev/CorrectionFactor, std_errv]) # Save Drift for each coordinate value
    #Pbar.finish()
    print('################### DONE ##########################')
    # np.savetxt('DQ.dat', DQ) # Save to file
    # np.savetxt('VQ.dat', VQ)
    # name for files
    filename = f.name + '.' + str(tmin) + '.' + str(tmax) + '.' + str(Qbins)

    DQ = np.asarray(DQ)
    VQ = np.asarray(VQ)

    np.savetxt('DQ' + filename + '.dat', DQ) # Save to file
    np.savetxt('VQ' + filename + '.dat', VQ)

    #to calculate F_{Stochastic}
    Z = np.stack((DQ[:,0], VQ[:,1]/DQ[:,1], np.sqrt(np.square(DQ[:,2])+np.square(VQ[:,2]))), axis=-1)
    Z = excludeinvalid(Z)
    W = np.stack((Z[:,0], integrate.cumtrapz(Z[:,1], Z[:,0], initial=Z[:,1][0]), Z[:,2]), axis=-1)
    W = excludeinvalid(W)
    G = np.empty(shape=[0,3])
    for Qi in DQ[:,0]:
        irow, icol = np.where(W == Qi)
        jrow, jcol = np.where(DQ == Qi)
        if (np.size(irow) != 0 and np.size(jrow) != 0):
            GQ = -(float(W[np.int(irow[0]), 1]))+np.log(float(DQ[np.int(jrow[0]), 1]))
            er = W[:,2][np.int(irow[0])]
        else:
            GQ = np.nan
            er = np.nan
        G = np.append(G, [[Qi, GQ, er]], axis=0)

    G = excludeinvalid(G)
    #print(G)

    SG = savgol_filter(G[:,1], 7, 3, mode='nearest')
    #Set minima related to folded state as zero
    idmin = argrelmin(SG)[-1][-1]
    G[:,1] = G[:,1]-G[:,1][idmin]


    np.savetxt('F_Stoch' + filename + '.dat', G)


    #Module to extract lines with errors
    diffusionfilename=str('DQ' + filename + '.dat') # Get the name of files
    vfilename=str('VQ' + filename + '.dat')
    freefilename=str('Free_energy_' + arg + '.dat')
    histfilename=str('hist_' + arg + '.dat')
    helmfilename=str('F_Stoch' + filename + '.dat')
    fn=[diffusionfilename, vfilename, freefilename, histfilename, helmfilename] #Make a list with filenames
    CheckFiles(fn)

    Qqzero = Q_zero
    Qqone = Q_one

    if Qqzero>Qqone: Qqzero, Qqone=Qqone, Qqzero # Must be Qqzero < Qqone

    ttaufold, uncerttaufold = calctau(beta, Qmin, Qqzero, Qqone, DQ, G)
    ttauunfold, uncerttauunfold = calctau(beta, Qmax, Qqone, Qqzero, DQ, G)
    ttTP, uncertttTP = calcmtpt(beta, Qqzero, Qqone, DQ, G)
    ttTPb, uncertttTPb = calcmtpt(beta, Qqone, Qqzero, DQ, G)


    #np.savetxt('pTPx_' + filename + '.dat', ptpx(beta, Qqzero, Qqone, DQ, G))

    ctAB, ctBA, cnTPAB, ctTPAB, cnTPBA, ctTPBA, cnAB, cnBA, cnTP, ctTP, cstdtAB, cstdtBA, cstdtTPAB, cstdtTPBA = calcttrajectory(Qqzero, Qqone, Q)


    print('mfpt calculated using Kramers equation from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(ttaufold) + ' +/- ' + str(uncerttaufold))
    print('mfpt calculated using Kramers equation from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(ttauunfold) + ' +/- ' + str(uncerttauunfold))
    print('mfpt measured using the trajectory from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctAB) + ' with ' + str(cnAB) + ' transitions.')
    print('mfpt measured using the trajectory from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(CorrectionFactor*ctBA) + ' with ' + str(cnBA) + ' transitions.')
    print('mtpt measured using the trajectory from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctTPAB) + ' with ' + str(cnTPAB) + ' transitions.')
    print('mtpt measured using the trajectory from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(CorrectionFactor*ctTPBA) + ' with ' + str(cnTPBA) + ' transitions.')
    print('Average mtpt measured using the trajectory between ' + str(Qqzero) + ' and ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctTP) + ' with ' + str(cnTP) + ' transitions.')
    print('mtpt calculated using Szabo equation for folding is ' + str(ttTP) + ' +/- ' + str(uncertttTP) + ' and for unfolding is '+ str(ttTPb) + ' +/- ' + str(uncertttTPb))


    matrix_m = np.empty(shape=[0,20])

    matrix_m = np.append( matrix_m, [['#mfpt-AB-Kramers', '#uncert-mfpt-AB-Kramers', 'mfpt-BA-Kramers', 'uncert-mfpt-BA-Kramers', 'mfpt-AB-trajectory', 'std-AB-trajectory', 'nAB', 'mfpt-BA-trajectory', 'std-BA-trajectory', 'nBA', 'average-mfpt', 'total-transitions', 'mtpt-AB-trajectory', 'std-mtpt-AB-trajectory', 'mtpt-BA-trajectory', 'std-mtpt-BA-trajectory', 'mtpt-AB-Szabo', 'uncert-mtpt-AB-Szabo', 'mtpt-BA-Szabo', 'uncert-mtpt-BA-Szabo']], axis=0)
    matrix_m = np.append( matrix_m, [[str(ttaufold), str(uncerttaufold), str(ttauunfold), str(uncerttauunfold), str(CorrectionFactor*ctAB), str(CorrectionFactor*cstdtAB), str(cnAB), str(CorrectionFactor*ctBA), str(CorrectionFactor*cstdtBA), str(cnBA), str(((CorrectionFactor*ctAB*cnAB+CorrectionFactor*ctBA*cnBA)/(cnAB+cnBA))), str((cnAB+cnBA)), str(CorrectionFactor*ctTPAB), str(CorrectionFactor*cstdtTPAB), str(CorrectionFactor*ctTPBA), str(CorrectionFactor*cstdtTPBA), str(ttTP), str(uncertttTP), str(ttTPb), str(uncertttTPb)]], axis=0)

    np.savetxt('mfpt-mtpt-' + filename + '.dat', matrix_m, fmt='%s')


    return

if __name__ == "__main__": main()
