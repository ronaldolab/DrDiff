#!/usr/bin/env python3
#usage pytest

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
import DrDiff as DrD

## Base file definitions

trajectory_reference = 'reaction_coordinate_Q'

DQ_reference = np.genfromtxt('DQreaction_coordinate_Q.2.6.1.dat')
VQ_reference = np.genfromtxt('VQreaction_coordinate_Q.2.6.1.dat')
Free_energy_reference = np.genfromtxt('Free_energy_reaction_coordinate_Q.dat')
F_Stoch_reference = np.genfromtxt('F_Stoch_reaction_coordinate_Q.2.6.1.dat')
old_rates = np.genfromtxt('mfpt-mtpt-reaction_coordinate_Q.2.6.1.dat', comments='#')


def finding_common_x_values(value, reference):
    """
    Function to compare two arrays and return the indexes for rows with common first column values
    Parameters
    ----------
    value, reference : ndarrays

    Returns
    -------
    idx_value, idx_reference : ndarrays of indexes
    """
    idx_value = np.asarray([np.where(a==value[:, 0])[0][0] for a in \
                            np.intersect1d(value[:, 0], reference[:, 0])])
    idx_reference = np.asarray([np.where(a==reference[:, 0])[0][0] for a in \
                            np.intersect1d(value[:, 0], reference[:, 0])])
    return idx_value, idx_reference

def compact_comparison(difference_array, input_array, threshold_zero=1e-298, threshold=1e-6):
    """
    Function to compare the difference_array values to threshold only where input_array values are above threshold_zero"""
    return np.less_equal(difference_array[np.less_equal(\
        np.absolute(input_array), threshold_zero)], threshold).all()


def comparison(value, reference):
    """
    Function to compare two vectors given a threshold and a percentage limit
    
    Parameters
    ----------
    value, reference : 1-D arrays.
    
    Returns
    -------
    out : boolean
        comparison between value and reference arrays.
    """
    # # To be used in the comparisons
    threshold = 1e-6 # (direct)
    threshold_per = 1e-4 # (percentage)
    # Will ignore percentage comparison if value or reference is below 1e-100
    threshold_zero = 1e-298
    # making sure both are numpy arrays
    value = np.asarray(value)
    reference = np.asarray(reference)# absolute difference between value and reference
    # Getting the reference of rows where first column values coincides
    idx_values, idx_reference = finding_common_x_values(value, reference)
    difference = np.absolute(np.subtract(value[idx_values], reference[idx_reference]))
    # direct comparison between the difference and threshold
    test_direct = np.less_equal(difference, threshold).all()
    # which values failed direct comparison (if any)
    idx_failed_direct = \
        np.where(np.less_equal(difference[:, 1], threshold)==False)[0]
    # # Keeping only lines where the direct comparison failed
    # value = value[idx_failed_direct]
    # reference =  reference[idx_failed_direct]
    # difference = difference[idx_failed_direct]
    if test_direct:
        print("Passed direct comparison.")
        test = test_direct
    else:
        # If none of the following tests return True, must return False
        test = False
        # Show where direct comparison is failing (if that happens)
        #print("{} failed direct comparison at lines {}.".format(idx_failed_direct.shape[0], idx_failed_direct))
        print("{} failed direct comparison.".format(idx_failed_direct.shape[0]))
        # Percent comparison.
        # First, evaluate the percentage difference on values where reference is not zero.
        percentage = np.divide(difference, np.absolute(reference), \
                               out=np.zeros_like(np.absolute(reference)), \
                                where=np.absolute(reference)!=0)
        idx_per = np.where(percentage[:, 1]>=threshold_per)[0]
        print("There are {} values above percentage threshold.".format(idx_per.shape[0]))
        # Then tests if direct difference is below threshold when either
        # correspondent value or reference is zero. Must be True to continue.
        if compact_comparison(difference, value, threshold_zero, threshold) \
            and compact_comparison(difference, reference, \
                threshold_zero, threshold):
            # Excluding percentage comparisons when value or reference is close
            # to zero.
            idx_value_is_far_from_zero = \
                np.where(np.greater_equal(np.absolute(value[:,4]), \
                                       threshold_zero))[0]
            idx_reference_is_far_from_zero = \
                np.where(np.greater_equal(np.absolute(reference[:,4]), \
                                       threshold_zero))[0]
            idx_is_far_from_zero = \
                np.unique(np.append(idx_value_is_far_from_zero, \
                                            idx_reference_is_far_from_zero))
            percentage = percentage[idx_is_far_from_zero]
            value = value[idx_is_far_from_zero]
            reference = reference[idx_is_far_from_zero]
            # In this test, we have to assure comparison only in the functions' results
            test_percentage = np.less_equal(percentage[:, 1], \
                                            threshold_per).all()
            if test_percentage:
                idx_passed = np.where(np.less_equal(percentage[:, 1], \
                                                    threshold_per)==True)[0]
                print("{} values have passed only by percentage comparison. In total {} values passed in the percentage comparison.".format(idx_failed_direct.shape[0], idx_passed.shape[0]))
            else:
                idx_percentage_fail = np.where(np.greater(percentage[:, 1], \
                                                          threshold_per))
                print("Largest percentage above threshold is {}.".format(percentage[:, 1].max()))
                print("Failed percentage comparison which values are {}.".format(percentage[:, 1][idx_percentage_fail]))
                print("Correspondent reference in the same places are {}".format(reference[:, 1][idx_percentage_fail]))
            test = test_percentage
        else:
            print("Failed because direct difference where one of the values is zero is above threshold.")
            test = False
    return test

def main_calculations(arg_traj):
    """Function to make the main calculations. Should be changed in the future"""
    
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
    
    f = open(arg_traj, 'r')
#    except (IOError) as errno:
#       print('I/O error. %s' % errno)
#       sys.exit()
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
    nbins = int(np.ceil((Qmax-Qmin)/Qbins))
    DrD.Free_energy_Histogram_Q(arg_traj, Q, nbins) ## Call function to Free Energy and Histogram
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
            DrD.Jump_Histogram(str(Qi) + '_' +str(t), y) ## Write Histograms with each t for each coordinate Qc
            ###############################################################################################
            #print(Qi, t, np.var(y), np.mean(y))
            D.append([t, 0.5*np.var(y)]) # Variance calculation - sigma^2
            V.append([t, np.mean(y)]) # Mean of histogram calculation - Qc

        D = np.asarray(D)
        V = np.asarray(V)
        sloped, interceptd, r_valued, p_valued, std_errd = stats.linregress(D[:, 0], D[:, 1]) # Linear regression - sloped is the difusion
        slopev, interceptv, r_valuev, p_valuev, std_errv = stats.linregress(V[:, 0], V[:, 1]) # Linear regression - slopev is the drift
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
    # Saving the files
    np.savetxt(filename + '_DQ.dat', DQ) # Save to file
    np.savetxt(filename + '_VQ.dat', VQ)

    #to calculate F_{Stochastic}
    Z = np.stack((DQ[:,0], np.divide(VQ[:,1], DQ[:,1]), np.sqrt(np.square(DQ[:,2])+np.square(VQ[:,2]))), axis=-1)
    Z =         (Z)
    W = np.stack((Z[:,0], integrate.cumulative_trapezoid(Z[:,1], Z[:,0], initial=0), Z[:,2]), axis=-1)
    W = DrD.excludeinvalid(W)
    G = np.empty(shape=[0,3])
    for Qi in DQ[:,0]:
        irow, icol = np.where(W == Qi)
        jrow, jcol = np.where(DQ == Qi)
        if (np.size(irow) != 0 and np.size(jrow) != 0):
            GQ = -(float(W[int(irow[0]), 1]))+np.log(float(DQ[int(jrow[0]), 1]))
            er = W[:,2][int(irow[0])]
        else:
            GQ = np.nan
            er = np.nan
        G = np.append(G, [[Qi, GQ, er]], axis=0)

    G = DrD.excludeinvalid(G)
    #print(G)

    SG = savgol_filter(G[:,1], 7, 3, mode='nearest')
    #Set minima related to folded state as zero
    idmin = argrelmin(SG)[-1][-1]
    G[:,1] = G[:,1]-G[:,1][idmin]

    np.savetxt(filename + '_F_Stoch.dat', G)

    Qqzero = Q_zero
    Qqone = Q_one

    if Qqzero > Qqone: Qqzero, Qqone = Qqone, Qqzero # Must be Qqzero < Qqone

    ttaufold, uncerttaufold = DrD.calctau(beta, Qmin, Qqzero, Qqone, DQ, G)
    ttauunfold, uncerttauunfold = DrD.calctau(beta, Qmax, Qqone, Qqzero, DQ, G)
    ttTP, uncertttTP = DrD.calcmtpt(beta, Qqzero, Qqone, DQ, G)
    ttTPb, uncertttTPb = DrD.calcmtpt(beta, Qqone, Qqzero, DQ, G)

    #np.savetxt('pTPx_' + filename + '.dat', ptpx(beta, Qqzero, Qqone, DQ, G))

    ctAB, ctBA, cnTPAB, ctTPAB, cnTPBA, ctTPBA, cnAB, cnBA, cnTP, ctTP, cstdtAB, cstdtBA, cstdtTPAB, cstdtTPBA = DrD.calcttrajectory(Qqzero, Qqone, Q)

    #print('mfpt calculated using Kramers equation from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(ttaufold) + ' +/- ' + str(uncerttaufold))
    #print('mfpt calculated using Kramers equation from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(ttauunfold) + ' +/- ' + str(uncerttauunfold))
    #print('mfpt measured using the trajectory from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctAB) + ' with ' + str(cnAB) + ' transitions.')
    #print('mfpt measured using the trajectory from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(CorrectionFactor*ctBA) + ' with ' + str(cnBA) + ' transitions.')
    #print('mtpt measured using the trajectory from ' + str(Qqzero) + ' to ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctTPAB) + ' with ' + str(cnTPAB) + ' transitions.')
    #print('mtpt measured using the trajectory from ' + str(Qqone) + ' to ' + str(Qqzero) + ' is ' + str(CorrectionFactor*ctTPBA) + ' with ' + str(cnTPBA) + ' transitions.')
    #print('Average mtpt measured using the trajectory between ' + str(Qqzero) + ' and ' + str(Qqone) + ' is ' + str(CorrectionFactor*ctTP) + ' with ' + str(cnTP) + ' transitions.')
    #print('mtpt calculated using Szabo equation for folding is ' + str(ttTP) + ' +/- ' + str(uncertttTP) + ' and for unfolding is '+ str(ttTPb) + ' +/- ' + str(uncertttTPb))

    matrix_m = np.empty(shape=[0,20])

    matrix_m = np.append( matrix_m, [['#mfpt-AB-Kramers', '#uncert-mfpt-AB-Kramers', 'mfpt-BA-Kramers', 'uncert-mfpt-BA-Kramers', 'mfpt-AB-trajectory', 'std-AB-trajectory', 'nAB', 'mfpt-BA-trajectory', 'std-BA-trajectory', 'nBA', 'average-mfpt', 'total-transitions', 'mtpt-AB-trajectory', 'std-mtpt-AB-trajectory', 'mtpt-BA-trajectory', 'std-mtpt-BA-trajectory', 'mtpt-AB-Szabo', 'uncert-mtpt-AB-Szabo', 'mtpt-BA-Szabo', 'uncert-mtpt-BA-Szabo']], axis=0)
    matrix_m = np.append( matrix_m, [[str(ttaufold), str(uncerttaufold), str(ttauunfold), str(uncerttauunfold), str(CorrectionFactor*ctAB), str(CorrectionFactor*cstdtAB), str(cnAB), str(CorrectionFactor*ctBA), str(CorrectionFactor*cstdtBA), str(cnBA), str(((CorrectionFactor*ctAB*cnAB+CorrectionFactor*ctBA*cnBA)/(cnAB+cnBA))), str((cnAB+cnBA)), str(CorrectionFactor*ctTPAB), str(CorrectionFactor*cstdtTPAB), str(CorrectionFactor*ctTPBA), str(CorrectionFactor*cstdtTPBA), str(ttTP), str(uncertttTP), str(ttTPb), str(uncertttTPb)]], axis=0)

    np.savetxt(filename + '_mfpt-mtpt.dat', matrix_m, fmt='%s')
    assert True

def test_load_traj():
    main_calculations(trajectory_reference)
    assert True

def test_diffusion():
    DQ = np.genfromtxt('reaction_coordinate_Q.2.6.1_DQ.dat')
    assert comparison(DQ, DQ_reference)

def test_drift():
    VQ = np.genfromtxt('reaction_coordinate_Q.2.6.1_VQ.dat')
    assert comparison(VQ, VQ_reference)

def test_fstoch():
    GStoch = np.genfromtxt('reaction_coordinate_Q.2.6.1_F_Stoch.dat')
    assert comparison(GStoch, F_Stoch_reference)

def test_rates():
    rates_current = np.genfromtxt('reaction_coordinate_Q.2.6.1_mfpt-mtpt.dat', comments='#')
    assert compact_comparison(rates_current, old_rates)