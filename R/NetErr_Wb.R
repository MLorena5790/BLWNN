# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .NetErr_Wb
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
# Purpose: Evaluation of the restricted Bayesian objective
#          function and extraction of unweighted error
#          components for Bayesian evidence procedures.
#
# Contribution: Developed for BLWNN based on the Bayesian
#               objective function structure of BLNN,
#               restricting regularization to conventional
#               network weights and biases while preserving
#               reparameterized wavelet parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

# --- Função de M(w) = β*E_D(w) + α*E_W(w) restrita aos parâmetros (W1, b1, W2, b2) ---
.NetErr_Wb  <-  function(Actual, NET, Evidence = TRUE)
  {
        # --- [1] Propagação direta (feedforward) com (t_tilde,log(s)) fixos
        fwded  <-  .ffwd(NET)                     
        yhat   <-  fwded[["trainout"]] 
         
        # --- [2] Cálculo do erro dos dados E_D(w) ---
        Err     <- if(fwded[["costF"]] == "crossEntropy") 
                        {
                          # Estabilização numérica para evitar log(0)
                          eps  <- .Machine$double.eps
                          yhat <- pmin(pmax(yhat, eps), 1 - eps)
                          
                          -sum(Actual * log(yhat) + (1 - Actual) * log(1 - yhat))
                          
                        }else{ mean((Actual - yhat)^2)} # Para MSE
                    
        
        
        # --- [4] Cálculo de M(w) = β*E_D(w) + α*E_W(w)
        FinalErr  <-   .ErrBayes(NET    = fwded, 
                                 NetErr = Err, 
                                 sep    = TRUE) 
        
        
        # --- [5] Apresentação do resultado 
        if(Evidence == TRUE)
        { 
          return(list(edata_pure  = Err,               # E_D(w) puro, sem ruído β
                      EVEP_Wb     = FinalErr$EVEP_Wb)) # ∑ E_W(w) puro separado por grupo W1, b1, W2, b2 
          
        }else{# Para otimização: M(w) = β*E_D(w) + α*E_W(w) p/ w = (W1, b1, W2, b2)
                Total   <-    FinalErr$edata + FinalErr$EP_Wb
                return(Total)
              }
  }
