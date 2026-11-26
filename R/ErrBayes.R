# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .ErrBayes
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
# Purpose: Evaluation of the Bayesian objective function
#          and its parameter-specific regularization terms.
#
# Contribution: Adapted and extended from the original BLNN
#               implementation to incorporate regularization
#               of reparameterized wavelet translation and
#               dilation parameters, including the decomposition
#               of penalty terms for Bayesian evidence procedures.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.ErrBayes   <-   function(
                          NET,          # objeto BLWNN
                          NetErr,       # Erros dos dados -> E_D 
                          sep = FALSE,  # maneira de exibição dos resultados 
                          ...)
{
  
    edata_p   <-   NetErr # E_D(w): erro dos dados puro 
  
  
    # --- [1] Cálculo de β.E_D ---
    # Se hp.Err > 0      --> β.E_D(w)
    # Se hp.Err = FALSE  --> E_D(w)
    if(is.numeric(NET[["scale.error"]]) && NET[["scale.error"]] > 0){
            edata  <-  edata_p * NET[["scale.error"]]
    } else {edata  <-  edata_p}
    

    # --- [2] Cálculo de E_W(w) = (1/2).wj² puro --- 
    # Priors (forma):
    #   W1       ~ N(0, 1)
    #   B1       ~ N(0, 1)
    #   W2       ~ N(0, 1)
    #   B2       ~ N(0, 1)
    #   t_tilde  ~ N(0, 1)
    #   log(r)   ~ N(0, 1)
    #
    # A escala (precisão) é introduzida posteriormente via:
    #   α_W1, α_B1, α_W2, α_B2, α_t_tilde, α_log.r
    w1_sq       <-  .5 * (NET[["weights"]][[1]]) ^ 2   # (1/2).W1² e (1/2).b1²
    w2_sq       <-  .5 * (NET[["weights"]][[2]]) ^ 2   # (1/2).W2² e (1/2).b2²
    t_tilde_sq  <-  .5 * (NET[["weights"]][[3]]) ^ 2   # (1/2).t_tilde²
    log.r_sq    <-  .5 * (NET[["weights"]][[4]]) ^ 2   # (1/2).log(r)²
  
    
    
    # --- [3]  ∑ E_W(w) puro separado por W1, b1, W2, b2 --- 
    # EXCLUSIVAMENTE p/ evidência bayesiana em:
    #                                           - Evidence == TRUE em .NetErr_Wb
    if (length(NET$scale.weights[[1]]) == 1) # α_W1 escalar → soma total de W1
    {                                        # ∑_j (1/2)·w1_j^2
      w1_sq_sum <- sum(w1_sq[1:NET$ncov, , drop = FALSE])
      
    } else { # ARD → α_W1 específico por covariável
             # (∑ (1/2)·w1_j^2) → soma por linha
              w1_sq_sum <- rowSums(w1_sq[1:NET$ncov, , drop = FALSE])
            }

    b1_sq_sum      <-  sum(w1_sq[NET$ncov + 1, , drop = FALSE])                 # ∑ (1/2)·b1²
    w2_sq_sum      <-  sum(w2_sq[1:(NET$hidden_size*NET$nout), , drop = FALSE]) # ∑_j (1/2)·w2_j²
    b2_sq_sum      <-  sum(w2_sq[(NET$hidden_size*NET$nout) + 1, ])             # ∑ (1/2)·b2²
    
    t_tilde_sq_sum <-  sum(t_tilde_sq[1:NET$hidden_size, ])          # ∑ (1/2)·t_tilde²
    log.r_sq_sum   <-  sum(log.r_sq[1:NET$hidden_size, ])            # ∑ (1/2)·log(r)²
    
    
    
    # --- [4] Multiplique α.E_W(w) = α.(1/2).wj² (força da regularização) --- 
    if(length(NET[["scale.weights"]][[1]]) == 1) # α_W1 é um único valor
      {  
          alpha_w1b1  <- c(rep(NET[["scale.weights"]][[1]], NET$ncov), # α_W1 (1)
                           NET[["scale.weights"]][[2]])                # α_b1 (1)
      } else { #  α_W1 específico p/ cda covariável (ARD)
                  alpha_w1b1  <- c(NET[["scale.weights"]][[1]],  # α_W1 (ncov)
                                   NET[["scale.weights"]][[2]])} # α_b1 (1)
    
      
    w1b1_sq_alpha    <-  sweep(w1_sq, 
                               1, 
                               alpha_w1b1,
                               FUN = "*")    # α_w1·(1/2)·w1²  
                                             # α_b1·(1/2)·b1²     
    
    alpha_w2b2       <-  c(rep(NET[["scale.weights"]][[3]], NET$hidden_size), # α_W2 (1)
                            NET[["scale.weights"]][[4]])                      # α_b2 (1)
    
    
    w2b2_sq_alpha    <-  sweep(w2_sq, 
                               1, 
                               alpha_w2b2, 
                               FUN = "*")   # α_w2·(1/2)·w2²  
                                            # α_b2·(1/2)·b2²                    
    
    t_tilde_sq_alpha <-  t_tilde_sq * NET[["scale.weights"]][[5]]  # α_t_tilde·(1/2)·t_tilde² 
    log.r_sq_alpha   <-  log.r_sq * NET[["scale.weights"]][[6]]    # α_log(r)·(1/2)·log(r)² 
    
    
    # --- [5] Soma total da regularização: ∑ α.E_W(w) --- 
    # p/ custo total: M(w) = β·E_D(w) + ∑ α·E_W(w)
    # (usada no treinamento completo (HMC / otimização))
    eprior_all   <-  sum(c( sum(w1b1_sq_alpha), 
                            sum(w2b2_sq_alpha),
                            sum(t_tilde_sq_alpha),
                            sum(log.r_sq_alpha)))
    
    
    # --- [6] Soma parcial da regularização: ∑ α.E_W(w) restrita a W1, b1, W2, b2--- 
    # EXCLUSIVAMENTE p/ evidência bayesiana, usada em: 
    #                                                 - fn1 de .BLWNN_BFGS_Wb
    #                                                 - Evidence == FALSE em .NetErr_Wb
    eprior_Wb   <-  sum(c( sum(w1b1_sq_alpha), 
                           sum(w2b2_sq_alpha)))
    
    # --- [7] Vetor ∑ E_W(w) puro separados por parâmetro W1, b1, W2, b2 --- 
    # Usado na evidência bayesiana p/:
    #                                 - Evidence == TRUE em .NetErr_Wb
    #                                 - cálculo de γ (effective number of parameters)
    eveprior_Wb    <-   c(w1_sq_sum, b1_sq_sum, w2_sq_sum, b2_sq_sum)
    
    # --- [7.1] Vetor ∑ E_W(w) puro separados por parâmetro W1, b1, W2, b2, t_tilde,log(r) --- 
    # Usado na evidência bayesiana p/:
    #                                 - sep == TRUE em .NetErr
    #                                 - cálculo de γ (effective number of parameters)
    eveprior_all   <-   c(w1_sq_sum, b1_sq_sum, w2_sq_sum, b2_sq_sum, t_tilde_sq_sum, log.r_sq_sum)

    
    # --- [8] Retorno do erro total ou decomposto por grupo ---
    if(sep == TRUE)                        
    { return(list(EP_all      = eprior_all,    # α.E_W(w) total: W1, b1, W2, b2, t_tilde, log(r)
                  edata       = edata,         # β.E_D(w)
                  EP_Wb       = eprior_Wb,     # α.E_W(w) p/ W1, b1, W2, b2
                  EVEP_Wb     = eveprior_Wb,  # E_W(w) puro por grupo W1, b1, W2, b2
                  EVEP_all    = eveprior_all))  # E_W(w) puro por grupo W1, b1, W2, b2, t_tilde, log(r)
      
    } else { return(edata + eprior_all)}       # M(w) = β.E_D(w) + α.E_W(w)

}
                          
                     
