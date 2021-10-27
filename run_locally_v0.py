#!/usr/bin/env python3
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
#   Vinicius de Godoi Contessoto:   vinicius.contessoto @gmail.com             #
#   Ronaldo Junio de Oliveira:      ronaldo.oliveira    @uftm.edu.br           #
#                                                                              #
#                                                                              #
# HOW TO RUN:                                                                  #
#                                                                              #
# python3 run_locally_v0.py TRAJECTORY                                         #
#                                                                              #
#                                                                              #
# Trajectory file:                                                             #
#                                                                              #
# Trajectories should be one-column time series, the reaction coordinate data  #
#                                                                              #
#                                                                              #
# Output files:                                                                #
#                                                                              #
# Outputs will be in the same dir starting with out-*                          #
#                                                                              #
# Python3 libraries are required:                                              #
# see file requirements_run_locally.txt                                        #
#                                                                              #
#                                                                              #
# Connect with us by email: drdiff.info@gmail.com                              #
#                                                                              #
#                   ##    GOOD LUCK    ##                                      #
#                                                                              #
################################################################################

import os
import sys
import DrDiff as drdiff

def main():

    if len(sys.argv) > 1: ## To open just if exist  file in argument
        FILENAME = sys.argv[1]
    else:
        print('No trajectory file found - Please insert a trajectory file')
        sys.exit()

    ## Global variables declaration

    Equilibration_Steps = 2000  # Equilibration Steps - Value to ignore the first X numbers from the traj file
    nbins = 100                 # Number of bins to be used while analyzing the trajectory
    tmax = 6                    # Default 6
    tmin = 2                    # Default 2
    time_step = 0.002           # time step value used to save the trajectory file - Default 0.001
    Snapshot = 5000             # Snapshots from simulation
    beta = 1                    # beta is 1/k_B*T
    Q_zero = 2.5                # Initial transition boundary
    Q_one = 4.2                 # Final transition boundary

    #working directory
    w_directory = str(os.getcwd()) + '/'

    result_times, out_traj, error_DrDiff = drdiff.do_calculation(('out-' + FILENAME + '-'), w_directory, (w_directory + FILENAME), (w_directory + 'out-' + FILENAME + '-'), beta, Equilibration_Steps, Q_zero, Q_one, nbins, time_step, Snapshot, tmin, tmax)


if __name__ == "__main__": main()
