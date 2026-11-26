# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .UpNetWbOnly
#
# Based on the original BLNN implementation:
# https://github.com/BLNNdevs/BLNN
#
# Author of the adaptation:
# Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
# (modifications and extensions)
#
# Developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Restricted reconstruction and update of
#          conventional network weights and biases.
#
# Contribution: Developed for BLWNN based on the parameter
#               reconstruction structure of BLNN, allowing
#               restricted parameter updates while preserving
#               reparameterized wavelet translation and
#               dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

# .UpNetWbOnly: Atualiza somente os pesos W1, W2 e os bias b1, b2 em um objeto BLWNN
.UpNetWbOnly  <-  function(wts,     # vetor unificado contendo W1, b1, W2, b2
                           NET)     # objeto BLWNN_Build a ser atualizado
{
  
  
  # ---[1] Vetor de novos valores ---
  NewWts    <-  wts 
  

  
  # ---[2] Pontos de quebra no vetor unificado ---
  expected  <-   (NET$ncov + 1) * NET$hidden_size +  # #(W1,B1)
                 (NET$hidden_size + 1) * NET$nout    # #(W2,B2)
  
  if(length(NewWts) != expected) {
    stop(sprintf("Esperado %d, recebido %d", expected, length(NewWts)))}
  
  
  # n. pesos de entrada  =  : (ncov + 1) * n_wavelons
  bkpt1    <-  (NET[["ncov"]] + 1) * NET[["hidden_size"]]
  # n. pesos de saída = : (n_wavelons + 1) * n_outputs
  bkpt2    <-  (NET[["hidden_size"]] + 1) * NET[["nout"]]
  
  
  
  # ---[3] Reconstrução de W1,b1 e W2,b2 ---
  w1_b1    <-  matrix(wts[1:bkpt1], 
                      ncol  = NET[["hidden_size"]], 
                      byrow = TRUE)
  
  w2_b2    <-  matrix(wts[(bkpt1 + 1):(bkpt1+bkpt2)], 
                      ncol  = NET[["nout"]], 
                      byrow = TRUE)
  
  
  # ---[4] Atualização de W1,b1 e W2,b2 na rede ---
  NET[["weights"]][[1]]   <-  w1_b1   # W1, b1 
  NET[["weights"]][[2]]   <-  w2_b2   # W2, b2
  
  
  # ---[5] Checagem defensiva ---
  if(!isTRUE(all.equal(c(t(NET$weights[[1]])),NewWts[1:bkpt1]))) {
    stop("Inconsistência em W1/b1 na reconstrução dos pesos (UpNetWts). Verifique c(t(.)) ↔ byrow=TRUE.")}
  
  if(!isTRUE(all.equal(c(t(NET$weights[[2]])),NewWts[(bkpt1 + 1):(bkpt1 + bkpt2)]))) {
    stop("Inconsistência em W2/b2 na reconstrução dos pesos (UpNetWts). Verifique c(t(.)) ↔ byrow=TRUE.")}
  
  
  return(NET)
  
}
