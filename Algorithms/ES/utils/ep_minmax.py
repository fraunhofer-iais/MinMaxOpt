#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mai 8  2020

Author: Aaron Klein, adapted to GPyOpt by Simon Bartels, abated to the min_max seting by Alexander Kister

The following functions are taken from
https://raw.githubusercontent.com/SheffieldML/GPyOpt/3bd165f349ae4bc37f5b675b4191c0eda5dac070/GPyOpt/util/epmgp.py
who are taken themselfs from 
https://github.com/automl/RoBO/blob/master/robo/util/epmgp.py
which seems to a be direct Matlab to Python translation of the code of
https://github.com/ProbabilisticNumerics/entropy-search
"""

#Copyright (c) 2015 Aaron Klein
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of RoBO nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from scipy import special

# some variables
sq2 = np.sqrt(2)
eps = np.finfo(np.float32).eps
l2p = np.log(2) + np.log(np.pi)


def joint_min_max(mu, var,nr_of_sim_parameters,proposed_worstcase, with_derivatives=False, idx2coords=None,**kwargs):
        """
    Computes the joint probability of 
        1)every given pair of simulation parameter and experiment number to be min_max 
        2)of the proposed_arg_max_function actually being the arg_max function (the function that returns, for each sim parameter, the worst case experiment)
    The Algo is based on the EPMGP[1] algorithm.
    [1] Cunningham, P. Hennig, and S. Lacoste-Julien.
    Gaussian probabilities and expectation propagation.
    under review. Preprint at arXiv, November 2011.

    Assumption the mu and var are the parameters of a Gaussian Process over a 2 D space. Application: One axis are the simulation paramters the other are the experiment parameters. 
    mu and var correspond to a list of  points in the 2 D space.
    The list is ordered by simulation parameter.
    Per simulation parmeter we have a equal amount of experiments.
    One of this entries is the proposed worst case, the variable proposed_worstcase indicates which one  the experimetns is the worst case for a given simulation parameter. 

 
    Parameters
    ----------
    M: np.ndarray(N,)
        Mean value of each of the N pairs between the simulation parameter idx and the experiment parameter.
        
    V: np.ndarray(N, N)
        Covariance matrix for all N pairs
    nr_of_sim_parameters: 
    proposed_worstcase: np.ndarray(N,)
        indicates the worst case experiment for each simulation parameter 
    with_derivatives: bool
        If true than also the gradients are computed 
    Returns
    -------
    np.ndarray(N,1)
        pmin distribution
    """

        logP = np.zeros(nr_of_sim_parameters)
        D = nr_of_sim_parameters
        D_with_experiments=len(proposed_worstcase)
        if with_derivatives:
            dlogPdMu = np.zeros((D, D_with_experiments))
            dlogPdSigma = np.zeros((D, int(0.5 * D_with_experiments * (D_with_experiments + 1))))
            dlogPdMudMu = np.zeros((D, D_with_experiments, D_with_experiments))
        for i in range(nr_of_sim_parameters):
                
                a = min_max_faktor(mu, var, i,nr_of_sim_parameters,proposed_worstcase)
                logP[i] = next(a)
                if with_derivatives:
                    dlogPdMu[i, :] = next(a).T
                    dlogPdMudMu[i, :, :] = next(a)
                    dlogPdSigma[i, :] = next(a).T
                    
        logP[np.isinf(logP)] = -500
        # re-normalize at the end, to smooth out numerical imbalances:
        logPold = logP
        Z = np.sum(np.exp(logPold))
        maxLogP = np.max(logP)
        s = maxLogP + np.log(np.sum(np.exp(logP - maxLogP)))
        s = maxLogP if np.isinf(s) else s # Better estimate of Z ?
        
        logP = logP - s
        
                
        if not with_derivatives:
                #Transform the jont dist of min_max and proposed_worstcase to a dist conditioned on the worst case
                logP = logP #Z=P(proposed_worstcase)
                return logP

                
        dlogPdMuold = dlogPdMu
        dlogPdSigmaold = dlogPdSigma
        dlogPdMudMuold = dlogPdMudMu
        # adjust derivatives, too. This is a bit tedious.
        Zm = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdMuold, 1)), 3)) / Z
        Zs = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdSigmaold, 1)), 3)) / Z
        
        dlogPdMu = dlogPdMuold - Zm
        dlogPdSigma = dlogPdSigmaold - Zs
        
        ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
        gg = np.einsum('kij,k->ij', dlogPdMudMuold + ff, np.exp(logPold)) / Z
        Zij = Zm.T * Zm
        adds = np.reshape(-gg + Zij, (1, D_with_experiments, D_with_experiments))
        dlogPdMudMu = dlogPdMudMuold + adds
        
        #Transform the jont dist of min_max and proposed_worstcase to a dist conditioned on the worst case
        #P'=P*(log P)'
        dZdMu=0#Maybe more correct: np.matmul(np.diag(np.exp(logP)),dlogPdMu)
        dZdSigma=0#Maybe more correct np.matmul(np.diag(np.exp(logP)),dlogPdSigma)
        dZdMudMu=0 #wrong: but we decided to ignor the derivative
        
        dlogZdMu=dZdMu
        dlogZdSigma=dZdSigma
                
        return logP, dlogPdMu-dlogZdMu, dlogPdSigma-dlogZdSigma, dlogPdMudMu-dZdMudMu


def factor_idx2compared_f_values(idx,k,nr_of_sim_parameters,proposed_worstcase):
        """p-rint('factor_idx2compared_f_values(idx,k,nr_of_sim_parameters,proposed_worstcase)')
        p-rint('proposed_worstcase {}'.format(proposed_worstcase))
        p-rint('len(proposed_worstcase) {}'.format(len(proposed_worstcase)))"""
        factor_checks_if_exp_is_the_arg_max_for_a_sim_par=False
        sim_idx=idx%nr_of_sim_parameters#idx pairs is order in row-major order 
        
        if proposed_worstcase[idx]>0:
                #the experiment is the proposed maximum,
                #the factor has to check if k is the minimum 
                l=k
                u=idx
        else:
                #the factor has to check if idx is the  smaller the the max for this simulation parameter
                l=idx
                #We search the corresponding claimed worst case experiment
                u=sim_idx
                while proposed_worstcase[u]==0:
                        u=u+nr_of_sim_parameters
                factor_checks_if_exp_is_the_arg_max_for_a_sim_par=True
        if l==u:
                l=None
                u=None
                
        return l,u, factor_checks_if_exp_is_the_arg_max_for_a_sim_par
    

def min_max_faktor(Mu, Sigma, k ,nr_of_sim_parameters,proposed_worstcase, gamma=1):
    #pri_nt('min_max_faktor(Mu, Sigma, k ,nr_of_sim_parameters,proposed_worstcase, gamma=1)')
    #pri_nt('proposed_worstcase:{}'.format(proposed_worstcase))
    D = Mu.size# There is one factor for each proposed point from the  2D plain

    #k is the idx of a sim param, it has to be mapped to the indx over pairs of sim param and exp param

    sim_idx_k=k#idx of pairs is order in row-major order
    u=sim_idx_k
    while proposed_worstcase[u]==0:
            u=u+nr_of_sim_parameters
    k=u#in pair index
    
    logS = np.zeros((D-1,))
    # mean time first moment
    MP = np.zeros((D-1,))
    
    # precision, second moment
    P = np.zeros((D-1,))
    
    M = np.copy(Mu)
    V = np.copy(Sigma)
    b = False
    d = np.NaN

    #Find the idx for which we have no idx because we compar k with k:
    expeption_idx=0
    for i in range(D):
            l,u , factor_chcks_for_arg_max= factor_idx2compared_f_values(i,k,nr_of_sim_parameters,proposed_worstcase)
            if l==None:
                    expeption_idx=i

    
    max_it=500
    start_phase_percent=0.0000
    for count in range(max_it):
        diff = 0
        #p-rint('count: {}, allready_past_l_eq_u: {}'.format(count,allready_past_l_eq_u))
        allready_past_l_eq_u=0
        for i in np.random.permutation(D):
            l,u , factor_chcks_for_arg_max= factor_idx2compared_f_values(i,k,nr_of_sim_parameters,proposed_worstcase)
            at_or_passt_expeption=(i>(expeption_idx-1))
            if count>max_it*start_phase_percent or factor_chcks_for_arg_max==False:
                    #Better for convergence to first fit to the argmin factors
                    if not l==None: #l==None is the expeption
                            try:
                                    idx=i-at_or_passt_expeption
                                    M, V, P[idx], MP[idx], logS[idx], d = lt_factor(l, u, M, V,
                                                                                    MP[idx], P[idx], gamma)
                            except Exception as e:
                                    print('Has this happend????????')
                                    raise
                                    
        
                            if np.isnan(d):
                                    break
                            diff += np.abs(d)
            
        final_bool=(count<max_it*start_phase_percent)
        if np.isnan(d):
                break
        if np.abs(diff) < 0.0000001 and final_bool:
                b = True
                break
    if count<max_it*start_phase_percent:
            print( RuntimeError("ep_minmax.py: Does not include all factors! count<max_it*0.1"))
    if np.isnan(d):
        
        #print('We are here because np.isnan(d)= True')
        logZ = -np.Infinity
        #print('ep_minmax.py before yield')
        yield logZ
        dlogZdMu = np.zeros((D, 1))
        yield dlogZdMu
    
        dlogZdMudMu = np.zeros((D, D))
        yield dlogZdMudMu
        dlogZdSigma = np.zeros((int(0.5 * (D * (D + 1))), 1))
        yield dlogZdSigma
        mvmin = [Mu[k], Sigma[k, k]]
        yield mvmin
    else:
        #print('We are here because np.isnan(d)= False')
        # evaluate log Z:
        C = np.eye(D) / sq2
        C[k, :] = -1 / sq2
        C = np.delete(C, k, 1)
        
        R = np.sqrt(P.T) * C
        r = np.sum(MP.T * C, 1)
        mp_not_zero = np.where(MP != 0)
        mpm = MP[mp_not_zero] * MP[mp_not_zero] / P[mp_not_zero]
        mpm = sum(mpm)
        
        s = sum(logS)
        IRSR = (np.eye(D - 1) + np.dot(np.dot(R.T, Sigma), R))
        rSr = np.dot(np.dot(r.T, Sigma), r)
        A = np.dot(R, np.linalg.solve(IRSR, R.T))
        
        A = 0.5 * (A.T + A)  # ensure symmetry.
        b = (Mu + np.dot(Sigma, r))
        Ab = np.dot(A, b)
        try:
            cIRSR = np.linalg.cholesky(IRSR)
        except np.linalg.LinAlgError:
            try:
                cIRSR = np.linalg.cholesky(IRSR + 1e-10 * np.eye(IRSR.shape[0]))
            except np.linalg.LinAlgError:
                    try:
                            cIRSR = np.linalg.cholesky(IRSR + 1e-6 * np.eye(IRSR.shape[0]))
                    except np.linalg.LinAlgError:
                            try:
                                    cIRSR = np.linalg.cholesky(IRSR + 1e-3 * np.eye(IRSR.shape[0]))
                            except np.linalg.LinAlgError:
                                    try:
                                            cIRSR = np.linalg.cholesky(IRSR + 1e-1 * np.eye(IRSR.shape[0]))
                                    except np.linalg.LinAlgError:
                                            try:
                                                    cIRSR = np.linalg.cholesky(IRSR + 1 * np.eye(IRSR.shape[0]))
                                            except np.linalg.LinAlgError:
                                                    cIRSR = np.linalg.cholesky(IRSR + 100 * np.eye(IRSR.shape[0]))
                                                    
                                                    
                                            
        dts = 2 * np.sum(np.log(np.diagonal(cIRSR)))
        logZ = 0.5 * (rSr - np.dot(b.T, Ab) - dts) + np.dot(Mu.T, r) + s - 0.5 * mpm
        #print('ep_minmax.py before yield')
        yield logZ
        btA = np.dot(b.T, A)
        
        dlogZdMu = r - Ab
        yield dlogZdMu
        dlogZdMudMu = -A
        yield dlogZdMudMu
        dlogZdSigma = -A - 2 * np.outer(r, Ab.T) + np.outer(r, r.T)\
                      + np.outer(btA.T, Ab.T)
        dlogZdSigma2 = np.zeros_like(dlogZdSigma)
        np.fill_diagonal(dlogZdSigma2, np.diagonal(dlogZdSigma))
        dlogZdSigma = 0.5 * (dlogZdSigma + dlogZdSigma.T - dlogZdSigma2)
        dlogZdSigma = np.rot90(dlogZdSigma, k=2)[np.triu_indices(D)][::-1]
        yield dlogZdSigma

                    
def lt_factor(s, l, M, V, mp, p, gamma):
        #pri_nt('s:{0}, l:{1}, M:{2}, V:{3}, mp:{4}, p:{5}, gamma:{6}'.format(s, l, M, V, mp, p, gamma))
        cVc = (V[l, l] - 2 * V[s, l] + V[s, s]) / 2.0
        Vc = (V[:, l] - V[:, s]) / sq2
        cM = (M[l] - M[s]) / sq2
        cVnic = np.max([cVc / (1 - p * cVc), 0])
        cmni = cM + cVnic * (p * cM - mp)
        z = cmni / np.sqrt(cVnic + 1e-25)
        if np.isnan(z):
                z = -np.inf
        e, lP, exit_flag = log_relative_gauss(z)
        if exit_flag == 0:
                alpha = e / np.sqrt(cVnic)
                # beta  = alpha * (alpha + cmni / cVnic);
                # r     = beta * cVnic / (1 - cVnic * beta);
                beta = alpha * (alpha * cVnic + cmni)
                r = beta / (1 - beta)
                # new message
                pnew = r / cVnic
                mpnew = r * (alpha + cmni / cVnic) + alpha
                
                # update terms
                dp = np.max([-p + eps, gamma * (pnew - p)])  # at worst, remove message
                dmp = np.max([-mp + eps, gamma * (mpnew - mp)])
                d = np.max([dmp, dp])  # for convergence measures
                
                pnew = p + dp
                mpnew = mp + dmp
                # project out to marginal
                
                Vnew = V - dp / (1 + dp * cVc) * np.outer(Vc, Vc)
                
                Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
                if np.any(np.isnan(Vnew)):
                        raise Exception("an error occurs while running expectation "
                                        "propagation in entropy search. "
                                        "Resulting variance contains NaN")
                # % there is a problem here, when z is very large
                logS = lP - 0.5 * (np.log(beta) - np.log(pnew) - np.log(cVnic))\
                       + (alpha * alpha) / (2 * beta) * cVnic
        
        elif exit_flag == -1:
                d = np.NAN
                Mnew = 0
                Vnew = 0
                pnew = 0
                mpnew = 0
                logS = -np.Infinity
        elif exit_flag == 1:
                d = 0
                # remove message from marginal:
                # new message
                pnew = 0
                mpnew = 0
                # update terms
                dp = -p  # at worst, remove message
                dmp = -mp
                d = max([dmp, dp])  # for convergence measures
                # project out to marginal
                Vnew = V - dp / (1 + dp * cVc) * (np.outer(Vc, Vc))
                Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
                logS = 0
        return Mnew, Vnew, pnew, mpnew, logS, d
                                                                                                                                                                                                                                                
def log_relative_gauss(z):
        """
        log_relative_gauss
        """
        if z < -6:
                return 1, -1.0e12, -1
        if z > 6:
                return 0, 0, 1
        else:
                logphi = -0.5 * (z * z + l2p)
                logPhi = np.log(.5 * special.erfc(-z / sq2))
                e = np.exp(logphi - logPhi)
        return e, logPhi, 0
