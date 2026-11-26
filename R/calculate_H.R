# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .calculate.H
#
# Original implementation: BLNN developers
# Source: https://github.com/BLNNdevs/BLNN
#
# Adaptations and modifications:
# Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
# (modifications and extensions)
#
# Developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Hamiltonian energy calculation for HMC
#          and NUTS sampling.
#
# Contribution: Adapted and modified from the original
#               BLNN implementation to accommodate
#               scalar, diagonal, and dense inverse
#               mass matrices within the BLWNN framework.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.calculate.H <- function(theta, r, fn, M_inv) {   U <- fn(theta)  # energia potencial
                                                  
                                                  # --- Caso 1: M_inv é escalar (massa identidade escalada)
                                                  if(is.numeric(M_inv) && length(M_inv) == 1) 
                                                    {K <- 0.5 * M_inv * sum(r^2)}
                                                  
                                                  # --- Caso 2: M_inv é vetor (massa diagonal)
                                                  else if(is.vector(M_inv)) 
                                                    {K <- 0.5 * sum(M_inv * r^2)}
                                                  
                                                  # --- Caso 3: M_inv é matriz densa
                                                  else if(is.matrix(M_inv)) 
                                                    {K <- 0.5 * as.numeric(crossprod(r, M_inv %*% r))}
                                                  
                                                  else {stop("M_inv must be scalar, vector (diagonal), or matrix.")}
                                                  
                                                  return(U + K)
                                              }
