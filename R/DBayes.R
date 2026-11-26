# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .DBayes
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
# Purpose: Computation of the Bayesian objective function
#          gradient by combining data-error gradients and
#          parameter-specific regularization gradients.
#
# Contribution: Adapted and extended from the original BLNN
#               implementation to incorporate regularization
#               gradients with respect to reparameterized
#               wavelet translation and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.DBayes   <-    function(NET,      # objeto .BLWNN_Build 
                         DatErr,   # RCweights -> (∂C/∂W1, ∂C/∂b1, ∂C/∂W2, 
                                   #               ∂C/∂b2, ∂C/∂t_tilde, ∂C/∂log(r))
                         ...)
{
  
  gdata_pure     <-  DatErr 
                     
  # β * ∇E_D(w)
  # Se hp.Err > 0      --> hp.Err * ∇E_D(w)
  # Se hp.Err = FALSE  --> ∇E_D(w)
  if(is.numeric(NET[["scale.error"]]) && NET[["scale.error"]] > 0){
          gdata  <-  lapply(gdata_pure, "*", NET[["scale.error"]]) 
  } else {gdata  <-  gdata_pure}
  
  
  
  # --- Cálculo de α * ∇E_W(w) = ∂/∂W(α * w^2/2) = α * w  ---
  if (length(NET[["scale.weights"]][[1]]) == 1) 
      { # hp.W1 é um único valor 
        alpha_w1b1 <- c(rep(NET[["scale.weights"]][[1]], NET$ncov), # α_w1 (1) (escalar)
                        NET[["scale.weights"]][[2]])                # α_b1 (1)
        
      } else { # hp.W1 específico p/ cda covariável
                alpha_w1b1 <- c(NET[["scale.weights"]][[1]],  # α_w1 (ncov) (vetor)
                                NET[["scale.weights"]][[2]])  # α_b1 (1)
      }
    # Validar a construção de α_w1, α_b1 
    stopifnot(length(alpha_w1b1) == nrow(NET[["weights"]][[1]]))
  
      # α_w1 * ∇E_W1 = ∂/∂W1(α_w1 * w1^2/2) = α_w1 * w1
      # α_b1 * ∇E_b1 = ∂/∂b1(α_b1 * b1^2/2) = α_b1 * b1
      w1b1_alpha  <-  sweep(NET[["weights"]][[1]], 
                          1, 
                          alpha_w1b1, 
                          FUN = "*")
  
      # Repetimos o mesmo hp.W2 para todos wavelons 
      # α_w2 * ∇E_W2 = ∂/∂W2(α_w2 * w2^2/2) = α_w2 * w2
      # α_b2 * ∇E_b2 = ∂/∂b2(α_b2 * b2^2/2) = α_b2 * b2
      alpha_w2b2 <- c(rep(NET[["scale.weights"]][[3]], NET$hidden_size),  # α_w2
                        NET[["scale.weights"]][[4]])                      # α_b2
      
      # Validar a construção de α_w2, α_b2 
      stopifnot(length(alpha_w2b2) == nrow(NET[["weights"]][[2]]))
    
      
      w2b2_alpha    <- sweep(NET[["weights"]][[2]], 
                           1, 
                           alpha_w2b2, 
                           FUN = "*")
      
      
      # Repetimos o mesmo hp.t.tilde para todos wavelons  
      # α_t * ∇E_t = ∂/∂t(α_t * t^2/2) = α_t * t
      t_tilde_alpha  <-  NET[["weights"]][[3]] * NET[["scale.weights"]][[5]] # α_t_tilde
      
      if(identical(NET$scale.weights[[5]], FALSE))
      t_tilde_alpha  <- 0 * NET$weights[[3]]
                                        
      
      # Repetimos o mesmo valor hp.r para todos wavelons   
      # α_log(r) * ∇E_log(r) = ∂/∂log(r)(α_log(r) * log(r)^2/2) = α_log(r) * log(r)
      # Onde: E_log(r) = (1/2) * log(r)^2 (energia do prior)
      log.r_alpha  <-  NET[["weights"]][[4]] * NET[["scale.weights"]][[6]]   # α_log.r   
      
      # Gradientes a priori => α * ∇E_W(w) 
      gprior  <-  list(w1b1_alpha, 
                       w2b2_alpha,
                       t_tilde_alpha,
                       log.r_alpha)
      
    
      # Gradiente da função de custo regularizada  ∂M(θ)/∂θ 
      # ∇M(w) = β * ∇E_D(w) + α * ∇E_W(w)
      BayesDeriv  <-  mapply("+", gdata, gprior)
      
      # Retorna lista de gradientes na mesma ordem de RCweights
      return(BayesDeriv)
    
  } 
