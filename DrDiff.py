# coding: utf8
################################################################################
#                                                                              #
# Calculate, using the Stochastic Drift-Diffusion Approach, the:               #
#                                                                              #
#   - Diffusion coefficient:                D(Q)                               #
#   - Free-Energy profiles:                 F_stochastic(Q), F_equilibrium(Q)  #
#   - Folding/unfolding times:              \tau_f, \tau_u                     #
#   - Transition Folding/unfolding times:   \tau_TS_f, \tau_TS_u               #
#                                                                              #
#                                                                              #
# Contribute by:                                                               #
#   Frederico Campos Freitas:       fredcfreitas        @gmail.com             #
#   Vinícius de Godoi Contessoto:   vinicius.contessoto @gmail.com             #
#   Ronaldo Junio de Oliveira:      ronaldo.oliveira    @uftm.edu.br           #
#                                                                              #
#                                                                              #
# HOW TO RUN:                                                                  #
#                                                                              #
# In the python3 environment, import the DrDiff class and execute              #
# the do_calculation method                                                    #
#                                                                              #
# python3.6                                                                    #
# from DrDiff import do_calculation                                            #
# do_calculation(userid)                                                       #
#                                                                              #
#                                                                              #
# Trajectory file:                                                             #
#                                                                              #
# Trajectories are stored in the trajectories folder in the main dir by naming #
# the file as userid_traj with just one column, the reaction coordinate data   #
#                                                                              #
#                                                                              #
# Output files:                                                                #
#                                                                              #
# Outputs will be in the output dir with files userid_*                        #
#                                                                              #
# Python3 libraries are required:                                              #
# numpy, scipy and itertools                                                   #
#                                                                              #
#                                                                              #
# Connect with us by email: drdiff.info@gmail.com                              #
#                                                                              #
#                   ##    GOOD LUCK    ##                                      #
#                                                                              #
################################################################################

import re
import os
import csv
#from itertools import islice
import numpy as np
from scipy import stats
from scipy import integrate
#from scipy.signal import argrelmin
#from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
#import plotly.graph_objs as go  # remove this and add matplotlib
#import plotly.io as pio         # remove this

################################################################################
# Method to clean the commentary lines from  the trajectory file               #
#                                                                              #
################################################################################
def clean_file(file):
    with open(file, "r") as f:
        data = f.read()
    with open("temporary", "w") as temp:
        temp.write(data)
    for pattern in ["^@.*|^%.*|^#.*|^;.*|^!.*|^[A-z].*|^$.*"]:
        matched = re.compile(pattern).search
        with open("temporary", "r") as temp:
            with open("temporary2", "w") as outfile:
                for line in temp:
                    if not matched(line):
                        outfile.write(line)
        os.replace(outfile.name, temp.name)
    return "temporary"

################################################################################

################################################################################
# Method to load a huge numpy matrix fastest than genfromtxt                   #
#                                                                              #
################################################################################
#A fastest way to read huge clean files
def read_large_txt(file, eq_steps=0, delimiter=None, dtype=None):
    with open(file, "r+") as FileObj:
        nrows = sum(1 for line in FileObj)
        FileObj.seek(0)
        ncols = len(next(FileObj).split(delimiter))
        out = np.empty((nrows, ncols), dtype=dtype)
        FileObj.seek(0)
        for i, line in enumerate(FileObj):
            out[i] = line.split(delimiter)
    return out[np.int(eq_steps):]

################################################################################

################################################################################
# Method to load the trajectory skipping the first Eq values                   #
#                                                                              #
################################################################################
def extract_trajectory(file, Eq):
    tempfile = clean_file(file)
    trajectory = read_large_txt(tempfile, Eq)
    try:
        os.remove(tempfile)
    except OSError:
        print("Error while deleting temporary file.")
    return trajectory.reshape(-1)

################################################################################


################################################################################
# Method to compare and return first array lines with same Q betwen both       #
#   F(Q) = - np.log [number-of-states(Q)]                                      #
################################################################################
def comparison_Q(first_array, second_array):
    compared = first_array[np.isin(first_array[:, 0], second_array[:, 0])]
    return compared
################################################################################


################################################################################
# Method to print Free Energy and the Histogram to data file                   #
#   F(Q) = - np.log [number-of-states(Q)]                                      #
################################################################################
def Free_energy_Histogram_Q(Qf, nbins, beta):

    #Histogram calculation
    hist, bins = np.histogram(Qf[::1], nbins, density=True)

    # Average of bins positions to plot
    bins_center = np.divide((bins[:-1] + bins[1:]), 2.0)

    # If you want to save the histogram file
    #np.savetxt(filename +  'hist.dat',np.c_[bins_center,hist])

    # Free Energy calculation
    FQ = np.divide(-np.log(hist), beta)
    Free = np.c_[bins_center, FQ]
    Free = excludeinvalid(Free)

    # Set minimum to zero + 1
    F0 = np.min(Free[:, 1])
    Free[:, 1] = Free[:, 1] - F0 + 1

    #If do you prefer a method in which the last minima is always zero:
    #FG = savgol_filter(FQ, 5, 3, mode='nearest')
    #Free = np.c_[bins_center, FQ]
    #id = argrelmin(FG)[-1][-1]
    #Free[:,1] = Free[:,1]-Free[:,1][id]

    return Free

################################################################################


################################################################################
# Method to print Histogram of t stes to data file                             #
#                                                                              #
################################################################################
def Jump_Histogram(filename, Qf):

    # Histogram calculation
    hist, bins = np.histogram(Qf[::1], density=True)

    # Average of bins positions to plot
    bins_center = np.divide((bins[:-1] + bins[1:]), 2.0)

    # Write Histogram file
    #np.savetxt('/Users/ronaldo/Sites/mysite/outputs/' + filename + '_H.dat',np.c_[bins_center,hist])

    return np.c_[bins_center, hist]

################################################################################

################################################################################
# Method to exclude invalid values for N-dimensional arrays                    #
################################################################################

def excludeinvalid(M):
    M = M[~np.isnan(M).any(axis=1)] #any(axis=-1) was returning error to 1D vectors.
    M = M[~np.isinf(M).any(axis=1)]
    M = M[~np.isneginf(M).any(axis=1)]
    return M
################################################################################

################################################################################
# Method to exclude invalid values for 1-D arrays                              #
################################################################################

def excludeinvalid1D(M):
    M = M[~np.isnan(M)]
    M = M[~np.isinf(M)]
    M = M[~np.isneginf(M)]
    return M
################################################################################

################################################################################
# Method to calculate D(Q) and v(Q)                                            #
#                                                                              #
################################################################################
def CalculateD_V(Q, Qmin, Qmax, Qbins, nbins, tmin, tmax, CorrectionFactor):

    DQ = np.empty(shape=(0, 3))
    VQ = np.empty(shape=(0, 3))

    # Vector just to avoid the empty end of file
    Add_end = np.linspace(Qmax, Qmax+Qbins, tmax + 1)

    Q = np.concatenate((Q, Add_end))

    bins        = np.linspace(Qmin, Qmax, nbins + 1)
    binscenter  = np.delete(np.divide((bins[:-1] + bins[1:]), 2.0), -1)

    # Calculate D and V coefficients as a function of Q
    for Qi in binscenter:

        # Find the Value of Qi with a bin in trajectory
        Q_index = np.array(np.where((Qi + Qbins > Q) & (Qi - Qbins < Q)))[0]

        D = np.empty(shape=(0, 2))
        V = np.empty(shape=(0, 2))
        #H = np.empty(shape=(0,2))

        # Loop of times for linear regression
        for t in range(tmin, tmax):

            y = Q[Q_index + t]

            #  Write Histograms to filename with each t for each coordinate Qc
            # H = np.append(H, [[Jump_Histogram(str(Qi) + '_' + str(t),y)]], axis=0)

            # Variance calculation: sigma^2
            D = np.append(D, [[t, np.multiply(np.var(y), 0.5)]], axis=0)

            # Mean of histogram calculation: Qc
            V = np.append(V, [[t, np.mean(y)]], axis=0)

        # Linear regression - sloped is the difusion
        sloped, interceptd, r_valued, p_valued, std_errd = stats.linregress(D)

        # Linear regression - slopev is the drift
        slopev, interceptv, r_valuev, p_valuev, std_errv = stats.linregress(V)

        # Save Diffusion for each coordinate value
        DQ = np.append(DQ, [[Qi, np.divide(sloped, CorrectionFactor), np.nan_to_num(np.divide(std_errd, sloped))]], axis=0)

        # Save Drift for each coordinate value
        VQ = np.append(VQ, [[Qi, np.divide(slopev, CorrectionFactor), np.nan_to_num(np.divide(std_errv, slopev))]], axis=0)


    DQ = np.asarray(DQ)
    VQ = np.asarray(VQ)

    DQ = excludeinvalid(DQ)
    VQ = excludeinvalid(VQ)

    return DQ, VQ


################################################################################
# Method to calculate Free Energy using D(Q) and v(Q)                          #
#   F_stochastic (Q) = G (Q)                                                   #
################################################################################
def Free_energy_Stochastic_Q(DQ, VQ, beta):
    DQ = comparison_Q(DQ, VQ)
    VQ = comparison_Q(VQ, DQ)
    #Old way to evaluate uncertainty
    Z = np.stack((DQ[:, 0], np.divide(VQ[:, 1], DQ[:, 1]), np.multiply(np.abs(np.divide(VQ[:, 1], DQ[:, 1])), np.sqrt(np.square(np.divide(DQ[:, 2], DQ[:, 1])) + np.square(np.divide(VQ[:, 2], VQ[:, 1]))))), axis=-1)
    Z = excludeinvalid(Z)
    W = np.stack((Z[:, 0], integrate.cumtrapz(Z[:, 1], Z[:, 0], initial=Z[:, 1][0]), Z[:, 2]), axis=-1)
    W = excludeinvalid(W)

    G = np.empty(shape=(0, 3))

    for Qi in DQ[:, 0]:

        irow, icol = np.where(np.equal(W, Qi))
        jrow, jcol = np.where(np.equal(DQ, Qi))

        if np.logical_and(np.not_equal(np.size(irow), 0), np.not_equal(np.size(jrow), 0)):

            GQ = -(float(W[np.int(irow[0]), 1])) + np.log(float(DQ[np.int(jrow[0]), 1]))
            GQ = np.divide(GQ, beta)
            er = np.multiply(np.abs(GQ), np.sqrt(np.square(np.divide(Z[:, 2][np.int(irow[0])], Z[:, 1][np.int(irow[0])])) + np.square(np.divide(DQ[:, 2][np.int(jrow[0])], DQ[:, 1][np.int(jrow[0])]))))

        else:

            GQ = np.nan
            er = np.nan

        G  = np.append(G, [[Qi, GQ, er]], axis=0)

    G = excludeinvalid(G)

    # Set minimum to zero
    G0 = np.min(G[:, 1])
    G[:, 1] = G[:, 1] - G0

    #If you want to assign zero for the last minimum
    #SG = savgol_filter(G[:,1], 7, 3, mode='nearest')
    #Set minima related to folded state as zero
    #idmin = argrelmin(SG)[-1][-1]
    #G[:,1] = G[:,1]-G[:,1][idmin]

    return G

################################################################################
# Method to calculate t_{folding} using Kramers equation                       #
#                                                                              #
################################################################################
def calctau(beta, Qinit, Qzero, Qone, DQ, G):

    utau    = 0
    DQ      = np.asarray(DQ)
    G       = np.asarray(G)
    DQ      = excludeinvalid(DQ)
    G       = excludeinvalid(G)
    tau     = np.empty(shape=(0, 3))
    taul    = np.empty(shape=(0, 3))

    # Get index of Q value close to Qzero
    idxzero = (np.abs(G[:, 0] - Qzero)).argmin()

    # Get index of Q value close to Qone
    idxone  = (np.abs(G[:, 0] - Qone)).argmin()

    if np.less(Qzero, Qone): x = 1
    else: x = -1

    # Summing from Qzero to Qone
    for Qj in G[:, 0][idxzero:idxone + x:x]:

        irow, icol   = np.where(DQ == Qj)
        jrow, jcol   = np.where(G == Qj)
        tau          = np.empty(shape=(0, 3))
        idxinit      = (np.abs(G[:, 0] - Qinit)).argmin()
        idxj         = (np.abs(G[:, 0] - Qj)).argmin()

        if np.less(Qinit, Qj): y=1
        else: y=-1

        # Summing from Qinit to Qj
        for Qk in G[:, 0][idxinit:idxj + y:y]:

            krow, kcol = np.where(G == Qk)

            if np.not_equal(np.size(irow), 0) and np.not_equal(np.size(jrow), 0) and np.not_equal(np.size(krow), 0):
                if not ((abs(float(G[np.int(jrow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1]) + abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(jrow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1]) - abs(3*np.nanstd(G, axis=0)[1])))):
                    GQ1 = (float(G[np.int(jrow[0]), 1]))
                    err1 = (float(G[np.int(jrow[0]), 2]))
                else:
                    GQ1 = np.nan
                    err1 = np.nan
                if not ((abs(float(G[np.int(krow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1]) + abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(krow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1]) - abs(3*np.nanstd(G, axis=0)[1])))):
                    GQ2 = (float(G[np.int(krow[0]), 1]))
                    err2 = (float(G[np.int(krow[0]), 2]))
                else:
                    GQ2 = np.nan
                    err2 = np.nan
                utau = np.divide((np.exp(np.multiply(beta, (GQ1 - GQ2)))), (float(DQ[np.int(irow[0]), 1]))) #calculating t_folding/unfolding
                if float(DQ[np.int(irow[0]), 1]) != 0:
                    uncerutau = np.multiply(np.absolute(utau), np.sqrt(np.multiply(np.square(beta), (np.square(err1) + np.square(err2))) + np.square(np.divide(float(DQ[np.int(irow[0]), 2]), float(DQ[np.int(irow[0]), 1])))))
                else:
                    uncerutau = 0
            else:
                utau = 0
                uncerutau = 0

            tau = np.append(tau, [[Qk, utau, uncerutau]], axis=0)
            tau = excludeinvalid(tau)

        #Inner integral
        inttau = integrate.cumtrapz(tau[:, 1], tau[:, 0], axis=0, initial=tau[0, 1])[-1] #inner integral
        uncertau = np.multiply(inttau, np.sqrt(np.mean(np.square(excludeinvalid1D(np.divide(tau[:, 2], tau[:,1])))))) #estimating error in inner integral
        taul = np.append(taul, [[Qj, inttau, uncertau]], axis=0)
        taul = excludeinvalid(taul)

    #Outer integral
    inttaul = integrate.cumtrapz(taul[:,1], taul[:,0], axis=0, initial=taul[0,1])[-1] #outer integral
    uncerttaul = inttaul*np.sqrt(np.amax(np.square(excludeinvalid1D(taul[:,2]/taul[:,1])))) #estimating error in inner integral
    #uncerttaul = np.multiply(inttaul, np.sqrt(np.amax(np.square(excludeinvalid1D(np.divide(taul[:,2], taul[:,1])))))) #estimating error in inner integral
    return inttaul, uncerttaul

################################################################################


################################################################################
# Method to calculate analytical mtpt                                          #
#                                                                              #
################################################################################
def calcmtpt(beta, Qzero, Qone, DQ, G):

    DQ = np.asarray(DQ)
    G = np.asarray(G)

    # Sampling left part of the integral
    vlint        = simpleint(testcalc, lcoreint, beta, Qzero, Qone, G, DQ)

    # Left integral from Qunf to Qfold
    intlintegral = integrate.cumtrapz(vlint[:, 1], vlint[:, 0], axis=0, initial=vlint[0, 1])[-1]

    # Sampling right part of integral
    vrint        = simpleint(testcalc, rcoreint, beta, Qzero, Qone, G, DQ)

    # Right integral from Qunf to Qfold
    intrintegral = integrate.cumtrapz(vrint[:, 1], vrint[:, 0], axis=0, initial = vrint[0, 1])[-1]

    inttpt       = np.multiply(intlintegral, intrintegral)

    #Using max uncertainty evaluated in both integral combinations as the final uncertainty
    np.seterr(divide='ignore', invalid='ignore')
    unmtpt = np.multiply(np.absolute(inttpt), np.sqrt(np.mean(np.square(excludeinvalid1D(np.divide(vlint[:, 2], vlint[:, 1])))) + np.mean(np.square(excludeinvalid1D(np.divide(vrint[:, 2], vrint[:, 1]))))))
    return inttpt, unmtpt

################################################################################


################################################################################
# Equation for mtpt - right integral                                           #
#                                                                              #
################################################################################
def rcoreint(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):
    # Calculate rintegral
    val = np.divide((np.exp(beta*(GQ1))), (float(DQ[np.int(irow[0]), 1])))
    #Using max uncertainty evaluated in both integral combinations as the final uncertainty
    if np.not_equal(float(DQ[np.int(irow[0]), 1]), 0):
        uncert = np.multiply(np.absolute(val), (np.sqrt(np.multiply(np.square(beta), np.square(unc)) + np.square(np.divide(float(DQ[np.int(irow[0]), 2]), float(DQ[np.int(irow[0]), 1]))))))
    else:
        uncert = 0
    return val, uncert

################################################################################


################################################################################
# Equation for mtpt - left integral                                            #
#                                                                              #
################################################################################
def lcoreint(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):

    xphi, unphi = phi(beta, Qzero, Qone, DQ, G, Qx)

    # Calculate lintegral
    val = (np.multiply(np.exp(np.multiply(np.multiply(-1, beta), (GQ1))), np.multiply(xphi, (1-xphi))))

    #Using max uncertainty as the final
    if np.logical_and(np.not_equal(GQ1, 0), np.not_equal(xphi, 0)):
        uncert = np.multiply(np.multiply(np.multiply(np.absolute(val), beta), np.absolute(np.multiply(np.multiply(np.multiply(np.multiply(-1, beta), (GQ1)), xphi), (1-xphi)))), (np.sqrt(np.square(np.divide(unc, GQ1))+np.square(np.divide(unphi, xphi)))))
    else:
        uncert = 0
    return val, uncert

################################################################################


################################################################################
# Method to \phi(x)                                                            #
#                                                                              #
################################################################################
def phi(beta, Qzero, Qone, DQ, G, qx):

    DQ          = np.asarray(DQ)
    G           = np.asarray(G)
    vlowphi     = simpleint(testcalc, equationphi, beta, Qzero, Qone, G, DQ)

    # Denominator integral from Qzero to Qone
    intlowphi   = integrate.cumtrapz(vlowphi[:, 1], vlowphi[:, 0], axis=0, initial=vlowphi[0, 1])[-1]
    vupphi      = simpleint(testcalc, equationphi, beta, Qzero, qx, G, DQ)

    # Numerator integral from Qzero to Q
    intupphi    = integrate.cumtrapz(vupphi[:, 1], vupphi[:, 0], axis=0, initial=vupphi[0, 1])[-1]

    phix        = np.divide(intupphi, intlowphi)
    #Using max uncertainty evaluated in both integral combinations as the final uncertainty
    uncphi = np.multiply(np.absolute(phix), np.sqrt(np.mean(np.square(excludeinvalid1D(np.divide(vupphi[:, 2], vupphi[:, 1])))) + np.mean(np.square(excludeinvalid1D(np.divide(vlowphi[:, 2], vlowphi[:, 1]))))))

    return phix, uncphi

################################################################################


################################################################################
# Method to evaluate values to a simple integral                               #
#                                                                              #
################################################################################
def simpleint(calctest, funcion, beta, Qzero, Qone, G, DQ):

    G = excludeinvalid(G)
    DQ = excludeinvalid(DQ)

    # Initializing two column numpy array
    sampledvalues = np.empty(shape=(0, 3))

    # Get index of Q value close to Qzero
    idxzero = (np.abs(G[:, 0] - Qzero)).argmin()

    # Get index of Q value close to Qone
    idxone  = (np.abs(G[:, 0] - Qone)).argmin()

    if Qzero < Qone: x=1
    else: x=-1

    # Summing from Qzero to Qone
    for Qx in G[:, 0][idxzero:idxone + x:x]:

        irow, icol = np.where(DQ == Qx)
        jrow, jcol = np.where(G == Qx)
        value, uncertainty = calctest(funcion, irow, jrow, G, DQ, beta, Qx, Qzero, Qone)
        sampledvalues = np.append(sampledvalues, [[Qx, value, uncertainty]], axis=0)

    #Excluding invalid values to
    sampledvalues = excludeinvalid(sampledvalues)

    return sampledvalues

################################################################################


################################################################################
# Method to calculate core of \phi(x) integral                                 #
#                                                                              #
################################################################################
def equationphi(irow, jrow, G, DQ, beta, GQ1, unc, Qx, Qzero, Qone):

    # Calculating phi core
    val = np.divide((np.exp(np.multiply(beta, (GQ1)))), (float(DQ[np.int(irow[0]), 1])))
    if float(DQ[np.int(irow[0]), 1]) != 0:
        uncert = np.multiply(np.absolute(val),
                            (np.sqrt(np.multiply(np.square(beta),
                            np.square(unc)) +
                            np.square(np.divide(float(DQ[np.int(irow[0]), 2]),
                            float(DQ[np.int(irow[0]), 1]))))))
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

    if np.size(irow) != 0 and np.size(jrow) != 0:
        if not ((abs(float(G[np.int(jrow[0]), 1])) >= (abs(np.nanmean(G, axis=0)[1])+abs(3*np.nanstd(G, axis=0)[1]))) or (abs(float(G[np.int(jrow[0]), 1])) <= (abs(np.nanmean(G, axis=0)[1])-abs(3*np.nanstd(G, axis=0)[1])))):
            GQ1  = (float(G[np.int(jrow[0]), 1]))
            unci = (float(G[np.int(jrow[0]), 2]))
        else:
            GQ1  = np.nan
            unci = np.nan
        eval, uncer = eq(irow, jrow, G, DQ, beta, GQ1, unci, Qx, Qzero, Qone)
    else:
        eval = 0
        uncer = 0
    return eval, uncer

################################################################################


################################################################################
# Method to calculate mfpt, mtpt and number of transitions from trajectory     #
#                                                                              #
################################################################################
def calcttrajectory(Qzero, Qone, Qtr):

    tAB = tBA = tTP = nAB = nBA = t0  = t1 = t2 = 0

    resultsAB  = np.empty(shape=(0, 4))
    resultsBA  = np.empty(shape=(0, 4))
    resultsTP0 = np.empty(shape=(0, 4))
    resultsTP2 = np.empty(shape=(0, 4))

    # Defines initial state. A is s==0
    if Qtr[0] <= Qzero: s = 0

    # B is s==2
    elif Qtr[0] >= Qone: s = 2

    # transition state
    else: s = 1

    for i in range(np.size(Qtr)):

        if s == 0:

            # identify last time when Q is lower than Q0
            if Qtr[i] <= Qzero: t1 = i + 1

            # identify when Q is greater than Q1
            if Qtr[i] >= Qone:
                t2 = i+1

                # count a transition
                nAB = nAB + 1
                s   = 2

                #add results in a row
                resultsAB  = np.append(resultsAB,  [[nAB, t0, t1, (t2-t0)]], axis=0)
                resultsTP0 = np.append(resultsTP0, [[nAB, t1, t2, (t2-t1)]], axis=0)
                t0 = t2
                t1 = t2

        elif s == 2:
            # identify last time when Q is greater than Q1
            if Qtr[i] >= Qone: t1 = i + 1

            # identify when Q is lower than Q0
            if Qtr[i] <= Qzero:
                t2 = i+1

                # count a transition
                nBA = nBA +1
                s = 0
                resultsBA = np.append(resultsBA, [[nBA, t0, t1, (t2-t0)]], axis=0)
                resultsTP2 = np.append(resultsTP2, [[nBA, t1, t2, (t2-t1)]], axis=0)
                t0 = t2
                t1 = t2
        elif s == 1:
            if Qtr[i + 1] <= Qzero: s = 0
            elif Qtr[i + 1] >= Qone: s = 2

    tAB      = np.divide((np.sum(resultsAB, axis=0)[3]), nAB)
    stdtAB   = np.nanstd(resultsAB, axis=0)[3]
    tBA      = np.divide((np.sum(resultsBA, axis=0)[3]), nBA)
    stdtBA   = np.nanstd(resultsBA, axis=0)[3]
    nTPAB    = (resultsTP0[(np.size(resultsTP0, axis=0) - 1)][0])
    tTPAB    = np.divide((np.sum(resultsTP0, axis=0)[3]), nTPAB)
    stdtTPAB = np.nanstd(resultsTP0, axis=0)[3]
    nTPBA    = (resultsTP2[(np.size(resultsTP2, axis=0) - 1)][0])
    tTPBA    = np.divide((np.sum(resultsTP2, axis=0)[3]), nTPBA)
    stdtTPBA = np.nanstd(resultsTP2, axis=0)[3]
    nTP      = (resultsTP0[(np.size(resultsTP0, axis=0) - 1)][0]) + (resultsTP2[(np.size(resultsTP2, axis=0) - 1)][0])
    tTP      = np.divide(((np.sum(resultsTP0, axis=0)[3]) + (np.sum(resultsTP2, axis=0)[3])), (nTP))

    #np.savetxt('Results.dat',resultsAB) #to print results matrix in a file
    return tAB, tBA, nTPAB, tTPAB, nTPBA, tTPBA, nAB, nBA, nTP, tTP, stdtAB, stdtBA, stdtTPAB, stdtTPBA

################################################################################


################################################################################
# Set of functions to evaluate p(TP|x)                                         #
#                                                                              #
################################################################################
def get_histogram(trajectory, dx, weights=None, dt=1):
    """
    Function to generate a histogram with dx as bin size, dt is the  \
    normalization value (if applicable).
    Return:
      sbins: upper bond value for each bin;
      svalues: the number of elements inside each bin definition, divided by dx.
    """
    trajectory = np.asarray(trajectory)
    sbins = np.arange(trajectory.min(), trajectory.max(), dx)
    svalues = np.divide(np.asarray(\
                [np.equal(np.digitize(trajectory[weights], sbins), x).sum() \
                for x in range(1, np.shape(sbins)[0] + 1)]), dx)
    return svalues, sbins


def get_state(trajectory, a, b):
    """Function to return the state for each frame, based on the following \
    criteria: below or equal a --> 0; between a and b --> 1; above or equal \
    b --> 2"""
    trajectory = np.asarray(trajectory)
    if a > b:
        a, b = b, a
    state = np.ones(shape=np.shape(trajectory)[0])
    state[np.less_equal(trajectory, a)] = 0
    state[np.greater_equal(trajectory, b)] = 2
    return state


def ret_changed_idx(state):
    """Function to return the indexes where the state has changed."""
    return np.where(np.not_equal(np.roll(state, 1), state))[0]


def get_transitions(state):
    """Function to return the transitions indexes, where the first value for \
    each row is when it starts and the second when it ends from the state \
    vector."""
    idx_changed = ret_changed_idx(state)
    if state[0] != 1:
        s = state[0]
        checked = []
    else:
        for i, j in enumerate(state):
            if j != 1:
                s = j
                checked = [i]
                break
    for i, j in enumerate(idx_changed):
        if state[j] == 2 and s != 2:
            s = 2
            checked.append([idx_changed[i-1], j])
        if state[j] == 0 and s != 0:
            s = 0
            checked.append([idx_changed[i-1], j])
    return np.asarray(checked)


def evaluate_ptpx(trajectory, a, b, dx=1, dt=1):
    """
    Function to evaluate the probability of being in a transition path \
    given the transition boundaries.
    Input:
      trajectory: 1-D trajectory (numpy.array)
      dx: bin size
      a, b: transition state boundaries.
    Output:
      p(TP|x): Nx2 numpy.array
    """
    trajectory = np.asarray(trajectory)
    #Vector with the successful transitions tagged with one.
    weight_transitions = np.zeros(shape=trajectory.shape[0])
    pairs_transitions = get_transitions(get_state(trajectory, a, b))
    for transition in pairs_transitions:
        weight_transitions[transition[0]: transition[1]] = 1
    # Transform weight_transitions to a boolean index array
    wvalues, wbins = get_histogram(trajectory, dx, \
                                   weights=weight_transitions.astype(bool), \
                                   dt=dt)
    avalues, abins = get_histogram(trajectory, dx, dt=dt)
    tvalues = np.zeros(shape=avalues.shape[0])
    fvalues = np.divide(wvalues, avalues, out=tvalues, \
                        where=np.not_equal(avalues, 0))
    ptpx_array = np.concatenate((abins.reshape(-1, 1), \
                                fvalues.reshape(-1, 1)), axis=1)
    return ptpx_array

################################################################################


################################################################################
# Plot the trajectory Q in a svg figure with matplotlib                        #
#                                                                              #
################################################################################
def plot_Q(Q, output_file_Q):

    # # Average over trajectory (# TODO: It is returning error:
    # #       101 libsystem_pthread.dylib             0x00007fff6978f40d thread_start
    # window_size = int(np.size(Q)/20)
    # window      = np.ones(window_size)/float(window_size)
    # Q_avg       = np.convolve(Q, window, 'same')

    t = np.linspace(0,np.size(Q),np.size(Q))

    # Save svg figure
    figure = plt.figure()
    fig, ax = plt.subplots()
    #ax.set_position([0, 0, 1, 1])
    ax.plot(t, Q)
    # ax.plot(t, Q_avg)
    ax.set(xlabel = 'time, t', ylabel = 'reaction coordinate, Q', title = 'Input trajectory file, Q(t)')
    ax.grid()
    fig.savefig(output_file_Q, bbox_inches=0)

    return

################################################################################

################################################################################
# Plot the trajectory Q in a svg figure with Plotly                            #
#                                                                              #
################################################################################
def plotly_Q(Q, output_file_Q, Q_zero, Q_one):

    #pio.orca.config.executable = '/usr/local/bin/orca'
    #plotly.io.orca.config.executable = '/usr/local/bin/orca'
    #pio.orca.config.save()

    # Average over trajectory
    #window_size = int(np.size(Q)/20)
    #window      = np.ones(window_size)/float(window_size)
    #Q_avg       = np.convolve(Q, window, 'same')
    t           = np.linspace(0, np.size(Q), np.size(Q))

    tmax = np.max(t)
    #tmin = np.min(t)
    t1 = [0, tmax]
    t_rev = t1[::-1]
    y_upper = [Q_one, Q_one]
    y_lower = [Q_zero, Q_zero]
    y_lower = y_lower[::-1]

    # Create a trace
    #TS
    trace1 = go.Scatter(x=t1+t_rev,
                        y=y_upper+y_lower,
                        fill='tozerox',
                        fillcolor='rgba(0,176,246,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='TS',
                        showlegend=True
                       )

    trace2 = go.Scatter(x=t,
                        y=Q,
                        line=dict(color='red'),
                        name='Q(t)',
                        showlegend=True
                       )

    #trace3 = go.Scatter(
    #    x = t,
    #    y = Q_avg,
    #    #line=dict(color='rgb(0,176,246)')
    #    line=dict(color='grey'),
    #    name='avg',
    #    showlegend=True
    #)

    #data = [trace1, trace2, trace3]
    data = [trace1, trace2]

    layout = go.Layout(title="Input trajectory file, Q(t)",
                       xaxis=dict(title='time, t', zeroline=True, showline=True,),
                       yaxis=dict(title='reaction coordinate, Q'),
                       legend=dict(x=0.9, y=1),
                      )

    fig = go.Figure(data=data, layout=layout)

    #pio.orca.status

    # Save svg figure
    pio.write_image(fig, output_file_Q)

    return

################################################################################

################################################################################
#                                                                              #
# Main method for the Stochastic Diffusion algorithm                           #
#                                                                              #
#   Trajectories are in the /tmp folder                                        #
#   Outputs will be in the /tmp output folder as runrid_*                      #
#                                                                              #
# this definition is for the form from the main.py website framework           #
#                                                                              #
################################################################################
def do_calculation(runId, path, filename, OUTPUT_FOLDER, beta, Eq, Q_zero, Q_one, Qbins, time_step, Snapshot, tmin, tmax):

    ## Global variables declaration
    # NOW SHOULD BE IN THE DO_CALCULATION DEFINITION
    # Receive from the main_page_form and output to an log file

    # Eq          = 10        # Equilibration Steps - Value to ignore the first X numbers from the traj file
    # Qbins       = 1         # Estimated bin width used to analyze the trajectory
    # tmax        = 6         # Default 6
    # tmin        = 2         # Default 2
    # time_step   = 0.0005    # time step value used to save the trajectory file - Default 0.001
    # Snapshot    = 50        # Snapshots from simulation
    # beta        = 1         # beta is 1/k_B*T
    # Q_zero      = 80        # Initial transition boundary
    # Q_one       = 230       # Final transition boundary

    CorrectionFactor = time_step*Snapshot

    out_folder = OUTPUT_FOLDER

    error = ''

    # Open trajectory file
    try:
        f = open(filename, 'r')
    except (IOError) as errno:
        error += 'I/O error. %s' % errno
        return (errno, error)

    #Using numpy.genfromtxt, it is unnecessary open the trajectory file. We just kept this part to check if there is some errors on it.
    Q = extract_trajectory(filename, Eq)

    # Save a figure in svg for the website
    #out_file_Q = path + '/static/trajectories/' + str(runId) + '_traj.svg'
    out_file_Q = out_folder + 'traj.svg'

    plot_Q(Q, out_file_Q)

    #plotly_Q(Q, out_file_Q, Q_zero, Q_one) # remove this

    # variable for img src in Flask static/trajectories/ folder
    #out_file_Q = 'trajectories/' + str(runId) + '_traj.svg'
    out_file_Q = str(runId) + '_traj.svg'

    # Take the max and min values
    Qmax = np.max(Q)
    Qmin = np.min(Q)

    # Test if Q_zero and Q_min were assigned properly. If not, it will be used random values
    if (Q_zero < Qmin) or (Q_one > Qmax):
        Q_zero = (Qmin + np.multiply(0.2, abs(Qmin)))
        Q_one = (Qmax - np.multiply(0.2, abs(Qmax)))
        #print('The transition state boundary was mischoosed. Your new transition state boundaries are = ', Q_zero, ' and ', Q_one)
    else:
        pass
        #print('The analysis will start.')
    # Number of bins for the reaction coordinate, Q
    nbins = np.int(np.ceil(np.divide((Qmax - Qmin), Qbins)))

    # Calculate Free Energy F(Q) from histogram (bins_center)
    Free = Free_energy_Histogram_Q(Q, nbins, beta)

    # Write FQ output file
    np.savetxt(out_folder + 'Free_energy.dat', Free)

    # Calculate D and v (Q)
    DQ, VQ = CalculateD_V(Q, Qmin, Qmax, Qbins, nbins, tmin, tmax, CorrectionFactor)

    # Save D and V files
    np.savetxt(out_folder + 'DQ.dat', DQ)
    np.savetxt(out_folder + 'VQ.dat', VQ)

    # Calculate F_{Stochastic}, as G here
    G = Free_energy_Stochastic_Q(DQ, VQ, beta)

    # Save F_st
    np.savetxt(out_folder + 'F_stoch_Q.dat', G)


    # Module to extract lines with errors (Should be revised because it will run in the server)
    # Get the name of files
    # diffusionfilename   = str(out_folder + 'DQ.dat')
    # vfilename           = str(out_folder + 'VQ.dat')
    # freefilename        = str(out_folder + 'Free_energy.dat')
    # histfilename        = str(out_folder + 'hist.dat')
    # helmfilename        = str(out_folder + 'F_stoch_Q.dat')
    #
    # # Make a list with filenames
    # fn = [diffusionfilename, vfilename, freefilename, histfilename, helmfilename]
    # CheckFiles(fn)

    # For the mfpt calculation
    Qqzero = Q_zero
    Qqone  = Q_one

    # Must be Qqzero < Qqone (Also verified in the input form section)
    if Qqzero>Qqone : Qqzero,Qqone = Qqone,Qqzero

    ttaufold    = calctau(beta, Qmin, Qqzero, Qqone, DQ, G)
    ttauunfold  = calctau(beta, Qmax, Qqone, Qqzero, DQ, G)
    ttTP        = calcmtpt(beta, Qqzero, Qqone, DQ, G)
    ttTPb       = calcmtpt(beta, Qqone, Qqzero, DQ, G)

    ctAB, ctBA, cnTPAB, ctTPAB, cnTPBA, ctTPBA, cnAB, cnBA, cnTP, ctTP, cstdtAB, cstdtBA, cstdtTPAB, cstdtTPBA = calcttrajectory(Qqzero, Qqone, Q)

    # Dictionary for times
    dict_times = {'mfpt_AB from Kramers equation' : ttaufold,
                  'mfpt_BA from Kramers equation' : ttauunfold,
                  'mfpt_AB from trajectory' : CorrectionFactor*ctAB,
                  'std_AB from trajectory' : CorrectionFactor*cstdtAB,
                  'number of transitions_AB' : cnAB,
                  'mfpt_BA from trajectory' : CorrectionFactor*ctBA,
                  'std_BA from trajectory' : CorrectionFactor*cstdtBA,
                  'number of transitions_BA' : cnBA,
                  'average mfpt' : CorrectionFactor*(ctAB*cnAB+ctBA*cnBA)/(cnAB+cnBA),
                  'total number of transitions' : cnAB+cnBA,
                  'mtpt_AB from trajectory' : CorrectionFactor*ctTPAB,
                  'std_mtpt_AB from trajectory' : CorrectionFactor*cstdtTPAB,
                  'mtpt_BA from trajectory' : CorrectionFactor*ctTPBA,
                  'std_mtpt_BA from trajectory' : CorrectionFactor*cstdtTPBA,
                  'mtpt_AB from Szabo equation' : ttTP,
                  'mtpt_BA from Szabo equation' : ttTPb
                 }

    # write dict_times to file
    timesfilename = str(out_folder + 'transition_times.csv')
    f_dict = open(timesfilename, "w")
    w = csv.writer(f_dict)

    for key, val in dict_times.items():
        w.writerow([key, val])

    f_dict.close()

    # write parameters input to a log file
    X = [beta, Eq, Q_zero, Q_one, Qbins, time_step, Snapshot, tmin, tmax]
    np.savetxt(out_folder + 'input_parameters.log', X, delimiter=',', fmt="%f",
               header="[beta, EquilibrationSteps, Q_zero, Q_one, Qbins, time_step, Snapshot, tmin, tmax]")


    return (dict_times, out_file_Q, error)

################################################################################
