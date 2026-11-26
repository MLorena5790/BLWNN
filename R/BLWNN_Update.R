# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: BLWNN_Update
#
# Original implementation: BLNN developers
# Source: https://github.com/BLNNdevs/BLNN
#
# Adaptations and extensions:
# Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
# (modifications and extensions)
#
# Developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Posterior-based update of Bayesian wavelet
#          neural network parameters and hyperparameters.
#
# Contribution: Adapted and extended from the original
#               BLNN implementation to accommodate the
#               extended parameter structure of BLWNN,
#               including reparameterized wavelet
#               translation and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

BLWNN_Update  <-  function(NET, fit, index=NULL)
      {
        sam    <- fit$sampler_params
        chains <- 1 #dim(sam)[2]  # n. de cadeias
        ite    <- dim(sam)[1]  # n. de iterações
        l      <- n.params #dim(sam)[3]  # n. de parâmetros 
        
        # Se não for especificado index. 
        if(is.null(index)) 
          index = fit$warmup+1
        
        if(chains == 1)
        {         # Calculando a média por coluna. 
          wts.up <- colMeans(sam[index:ite, , ])[-l]
        }else {
          wts.up <- colMeans(colMeans(sam[index:ite, , ]))[-l]
        }
        
        # Substituindo os parâmetros da rede no objeto BLWNN_Build 
        Net.up                <- .UpNetWts(wts.up, NET)
        
        # Lista contendo hiperparâmetros 
        # (hp.w1, hp.b1, hp.w2, hp.b2, hp.t, hp.s)
        Net.up$scale.weights  <- fit$hp.list$hp.W
        
        # Lista contendo o hiperparâmetro de erro hp.er
        Net.up$scale.error    <- fit$hp.list$hp.E
        
        return(Net.up)
      }

