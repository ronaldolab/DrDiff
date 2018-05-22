#!/usr/bin/env python
#coding: utf8
########################################################################
#
# Calculate the Diffusion Coefficient (D(Q)) and the Free-Energy (G(Q))
#   using the Stochastic Approach
#
#
# Contribute by:
#  Vinícius de Godoi Contessoto
#  Frederico Campos Freitas
#  Ronaldo Junio de Oliveira
#
#
# python StochasticDiffusion.py trajectory_file
#
#  trajectory (just one column - coordinate)
#
#
# PS: Need to install some libraries: numpy, scipy and itertools
#
#######################################################################

import sys
import numpy as np
import scipy as sc
#import progressbar
#from time import sleep
from itertools import islice
from scipy import stats
import fnmatch as fm
import os

##################################################################################################
# Function to print Free Energy and the Histogram to data file
#   F(Q) = - np.log [number-of-states(Q)]
##################################################################################################
def Free_energy_Histogram_Q(filename,Qf,Qbins):

   #Histogram calculation
   hist,bins=np.histogram(Qf[::1],int(np.ceil(np.max(Qf)/Qbins)),density=True) ##
   #hist,bins=np.histogram(Qf[::1],28,density=True)
   bins_center = (bins[:-1] + bins[1:])/2.0 ## Average of bins positions to plot
   #np.savetxt('hist_' + filename + '.dat',np.c_[bins_center,hist]) ## Write Histogram file
   FQ = -np.log(hist) ## Free Energy calculation
   np.savetxt('Free_energy_' + filename + '.dat',np.c_[bins_center,FQ]) ## Write Free Energy file
   #print('Coordinate Histogram and Free Energy calculated')
   #print '################################################'
   return
##################################################################################################

##################################################################################################
# Function to print Histogram of t stes to data file
#
##################################################################################################
def Jump_Histogram(filename,Qf):

   hist,bins=np.histogram(Qf[::1],density=True) ## Histogram calculation
   bins_center = (bins[:-1] + bins[1:])/2.0 ## Average of bins positions to plot
   #np.savetxt('H_' + filename + '.dat',np.c_[bins_center,hist]) ## Write Histogram file
   return
##################################################################################################


##################################################################################################

def CheckFiles(q):
    for aw in os.listdir('.'):
        for w in q: # Delete lines with "nan" and "inf" inside.
            if aw==w:
                ff = open(w,"r+")
                dd = ff.readlines()
                ff.seek(0)
                for z in dd:
                    if (("nan" not in z) and ("inf" not in z)):
                        ff.write(z)
                ff.truncate()
                ff.close()
    print '################## CHECKED ########################'
return

##################################################################################################


## Global variables declaration

Eq=10 # Equilibration Steps - Value to ignore the first X numbers from the traj file
Qbins=1.5 # bins read from trajectory - Default Qbins=1 to proteins
tmax=6 # Default 6
tmin=2 # Default 2
time_step=1 # time step value used to save the trajectory file - Default 0.001
Snapshot=1 # Snapshots from simulation
CorrectionFactor = time_step*Snapshot

#print '################################################'
print 'Equilibration Steps =',Eq
print 'Bins read from trajectory =',Qbins
print 'Time Step =',time_step
print 'Snapshot =',Snapshot
print 'tmax =',tmax,'| tmin =',tmin
#print '################################################'

def main():

   if len(sys.argv) > 1: ## To open just if exist  file in argument
     arg = sys.argv[1]
   else:
     print ('No trajectory file found - Please insert a trajectory file')
     sys.exit()
   try:
     f = open(arg, 'r')
   except (IOError) as errno:
     print ('I/O error. %s' % errno)
     sys.exit()
   print 'Reading trajectory file'
   #print '################################################'
   #Pbar = progressbar.ProgressBar(term_width=53,widgets=['Working: ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
   Q = np.asarray([float(line.rstrip()) for line in islice(f, Eq, None)]) # Save the coordinate value skipping the Equilibration steps
   Qmax = np.int(np.max(Q)) # take the max and min value
   Qmin = np.int(np.min(Q))
   print 'From trajectory file'
   print 'Qmax =',Qmax,'| Qmin =',Qmin
   print 'Mean(Q) =',np.mean(Q),'| Std(Q) =',np.std(Q)
   print 'Std(Q)/Mean(Q) =',np.std(Q)/np.mean(Q)
   #print '################################################'
   #Qmax=1.0
   Free_energy_Histogram_Q(arg,Q,Qbins) ## Call function to Free Energy and Histogram
   DQ=[]
   VQ=[]
   Add_end=np.arange(Qmax,Qmax+tmax+1) ## Vector just to avoid the empty end of file
   #Add_end=np.linspace(Qmax,Qmax+Qbins,tmax+1)
   Q = np.concatenate((Q,Add_end))
   #Pbar.start()
   for Qi in np.arange(Qmin+1.0,Qmax-1.0,Qbins):
      Q_index = np.array(np.where( (Qi + Qbins >= Q) & (Qi - Qbins <= Q)))[0] ## Find the Value of Qi with a bin in trajectory
      x=Q[Q_index]
      D=[]
      V=[]
      #Pbar.update( (Qi-Qmin)*100/(Qmax-Qmin))
      #sleep(0.01)
      for t in range(tmin,tmax): # Loop of times for linear regression
         #print Q_index+t,t,Q[Q_index]
         y=Q[Q_index+t]
         ####################  Print Historgram Do not delete it #######################################
         Jump_Histogram(str(Qi) + '_' +str(t),y) ## Write Histograms with each t for each coordinate Qc
         ###############################################################################################
         #print Qi,t,np.var(y),np.mean(y)
         D.append([t,0.5*np.var(y)]) # Variance calculation - sigma^2
         V.append([t,np.mean(y)]) # Mean of histogram calculation - Qc

      sloped, interceptd, r_valued, p_valued, std_errd = stats.linregress(D) # Linear regression - sloped is the difusion
      slopev, interceptv, r_valuev, p_valuev, std_errv = stats.linregress(V) # Linear regression - slopev is the drift
      DQ.append([Qi,sloped/CorrectionFactor]) # Save Diffusion for each coordinate value
      VQ.append([Qi,slopev/CorrectionFactor]) # Save Drift for each coordinate value
   #Pbar.finish()
   print '################## DONE ########################'
#   np.savetxt('DQ.dat',DQ) # Save to file
#   np.savetxt('VQ.dat',VQ)
   # name for files
   filename = f.name + '.' + str(tmin) + '.' + str(tmax) + '.' + str(Qbins)

   np.savetxt('DQ' + filename + '.dat',DQ) # Save to file
   np.savetxt('VQ' + filename + '.dat',VQ)

   #print VQ
   X=np.asarray([i[1] for i in DQ])
   Y=np.asarray([i[1] for i in VQ])
   Z=Y/X
   W=np.cumsum(Z)
   #print W,W[1],Z
   G=[]
   i=0
   for Qi in np.arange(Qmin+1.0,Qmax-1.0,Qbins):

       G.append([Qi,(-W[i]+np.log(X[i]))])
       #print Qi,W[i],i,np.log(X[i])
       i=i+1
   #print G
   np.savetxt('GQ' + filename + '.dat',G)

   #Module to extract lines with errors
   diffusionfilename=str('DQ' + filename + '.dat') # Get the name of files
   vfilename=str('VQ' + filename + '.dat')
   freefilename=str('Free_energy_' + arg + '.dat')
   histfilename=str('hist_' + arg + '.dat')
   gibbsfilename=str('GQ' + filename + '.dat')
   fn=[diffusionfilename, vfilename, freefilename, histfilename,gibbsfilename] #Make a list with filenames
   CheckFiles(fn)


   return

if __name__ == "__main__": main()
