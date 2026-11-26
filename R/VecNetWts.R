# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .VecNetWts
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
# Purpose: Vectorization and concatenation of all network
#          parameters into a single numerical vector.
#
# Contribution: Adapted and extended from the original BLNN
#               implementation to accommodate reparameterized
#               wavelet translation and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.VecNetWts   <-  function(NET)
      {
          w1       <-  c(t(NET[["weights"]][[1]]))   # W1 + b1
          w2       <-  c(t(NET[["weights"]][[2]]))   # W2 + b2
          t_tilde  <-  c(t(NET[["weights"]][[3]]))   # t_tilde
          log.r    <-  c(t(NET[["weights"]][[4]]))   # log(r), onde r = 1/s (dilatação inversa)
          
          # Vetor completo de todos os parâmetros 
          VecW  <-  c(w1, w2, t_tilde, log.r)
          return(VecW)
      }
