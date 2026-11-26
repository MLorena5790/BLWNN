# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .compute_next_window
# Author: Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
#
# Original function developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Computation of the next metric adaptation
#          window endpoint during warmup.
#
# Contribution: Originally developed for BLWNN to support
#               progressively increasing adaptation windows
#               and adjustment of the final metric
#               adaptation window.
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.compute_next_window <- function(i, anw, warmup, w1, aws, w3)
{
  anw <- i + aws
  if(anw == (warmup - w3)) return(anw)
  nwb <- anw + 2 * aws
  
  if(nwb >= warmup - w3){anw <- warmup - w3}
  return(anw)
}
