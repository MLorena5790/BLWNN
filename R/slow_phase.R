# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .slow_phase
# Author: Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
#
# Original function developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Identification of the slow adaptation phase
#          during the warmup period.
#
# Contribution: Originally developed for BLWNN to support
#               window-based metric adaptation during
#               Hamiltonian Monte Carlo sampling.
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.slow_phase <- function(i, warmup, w1, w3)
{
  x1 <- i >= w1
  x2 <- i <= (warmup - w3)
  x3 <- i < warmup
  return(x1 & x2 & x3)
}
