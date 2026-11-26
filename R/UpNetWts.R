# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .UpNetWts
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
# Purpose: Reconstruction and update of network parameters
#          from a unified numerical parameter vector.
#
# Contribution: Adapted and extended from the original BLNN
#               implementation to accommodate reparameterized
#               wavelet translation and dilation parameters,
#               including additional consistency checks.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.UpNetWts  <-  function(wts,     # vetor de parâmetros unificado (.VecNetWts)
                        NET)     # objeto BLWNN_Build a ser atualizado
{
    # ---[1] Vetor de novos valores ---
    NewWts    <-  wts 
    
    
    # ---[2] Quebras para divisão do vetor dos parâmetros ---
    expected  <-   (NET$ncov + 1) * NET$hidden_size +  # #(W1,B1)
                   (NET$hidden_size + 1) * NET$nout +  # #(W2,B2)
                   NET$hidden_size +                   # #(t_tilde)
                   NET$hidden_size                     # #(log.r)
    
    if(length(NewWts) != expected) {
      stop(sprintf("Esperado %d, recebido %d", expected, length(NewWts)))}
    
    

    # n. pesos de entrada  =  : (ncov + 1) * n_wavelons
    bkpt1    <-  (NET[["ncov"]] + 1) * NET[["hidden_size"]]
    
    # n. pesos de saída = : (n_wavelons + 1) * n_outputs
    bkpt2    <-  (NET[["hidden_size"]] + 1) * NET[["nout"]]
    
    # n. t_tilde (1 por wavelon)
    bkpt3    <-  NET[["hidden_size"]]
    
    
    # ---[3] Reconstrução dos pesos a partir do vetor unificado ---
    w1_b1    <-  matrix(NewWts[1:bkpt1], 
                         ncol  = NET[["hidden_size"]], 
                         byrow = TRUE)
    
    w2_b2    <-  matrix(NewWts[(bkpt1 + 1):(bkpt1+bkpt2)], 
                         ncol  = NET[["nout"]], 
                         byrow = TRUE)
    
    t_tilde  <-  matrix(NewWts[((bkpt1+bkpt2) + 1):((bkpt1+bkpt2) + bkpt3)], 
                        ncol  = 1)
    
    log.r    <-  matrix(NewWts[((bkpt1+bkpt2+bkpt3) + 1):length(NewWts)], 
                        ncol  = 1)
    
    # ---[4] Atualização dos pesos no objeto da rede ---
    NET[["weights"]][[1]]   <-  w1_b1    #atualizando W1 E b1 (NewWts)
    NET[["weights"]][[2]]   <-  w2_b2    #atualizando W2 e b2 (NewWts)
    NET[["weights"]][[3]]   <-  t_tilde  #atualizando t_tilde (NewWts)
    NET[["weights"]][[4]]   <-  log.r    #atualizando log(r) (r = 1/s) (NewWts)
    
    
    # ---[5] Checagem defensiva ---
    if(!isTRUE(all.equal(c(t(NET$weights[[1]])),NewWts[1:bkpt1]))) {
      stop("Inconsistência em W1/b1 na reconstrução dos pesos (UpNetWts). Verifique c(t(.)) ↔ byrow=TRUE.")}
    
    if(!isTRUE(all.equal(c(t(NET$weights[[2]])),NewWts[(bkpt1 + 1):(bkpt1 + bkpt2)]))) {
      stop("Inconsistência em W2/b2 na reconstrução dos pesos (UpNetWts). Verifique c(t(.)) ↔ byrow=TRUE.")}
    
    
    return(NET)

}



