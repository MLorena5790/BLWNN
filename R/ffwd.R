# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .ffwd
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
# Purpose: Forward propagation through the hidden and
#          output layers of Bayesian wavelet neural networks.
#
# Contribution: Adapted and extended from the original BLNN
#               implementation to incorporate wavelet
#               activation functions and reparameterized
#               translation and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.ffwd  <-  function(
                        NET,  # objeto BLWNN -> carrega características principais da rede
                    ...)
{

  # Forward da camada oculta:
  # a = (W1 * x + b1).r - t_tilde
  # h = ψ(a)
                        
  func  <-  NULL
  
  # --- Funções wavelets ψ(a) ---
  #___________________________________________________________________________________________________
  if(NET[["waveF"]] == "sig_dev2")    func  <- bquote({
                                                          sig  <- 1 / (1 + exp(-a))       
                                                          sig * (1 - sig) * (1 - 2*sig)  #ψ(sig) = 2sig³ - 3sig² + sig - 
                                                        })
  
  if(NET[["waveF"]] == "mexicanhat")  func  <-  bquote({ (1-a**2) * exp(-a**2/2)}) 
  #___________________________________________________________________________________________________
  
      # matriz de valores u = (w1*x + b1) 
      u   <-  cbind(NET[["normData"]], 1) %*% (NET[["weights"]][[1]])
      
      # armazenando u 
      NET[["wavePre"]]  <-  u  
      
      # cálculo de r = exp(log.r) ---
      # OBS: r = 1/s, onde s era a dilatação original
      # Nova parametrização: a = u * r - t_tilde (mais estável para HMC)
      r      <-  exp(NET[["weights"]][[4]]) 
      
      #if(any(!is.finite(r)) || any(r <= 0) || any(r > 1e6))
      if(any(!is.finite(r)))
        stop("Invalid inverse-scale encountered")
      
      # calculando a matriz de valores a = u.r - t_tilde 
      a               <-  sweep(sweep(u, 
                                      2, 
                                      r, 
                                      FUN="*"), 
                                2, 
                                NET[["weights"]][[3]], 
                                FUN="-")
      
      
      # armazenando u.r - t_tilde
      NET[["waveInput"]]  <-  a
      
      # calculando a matriz de valores h = ψ(a)
      h                   <-  eval(func)
      
      # armazenando h = ψ(a) 
      NET[["waveOut"]]    <-  h  
      

      
      
  # Forward da camada saída:
  # z = (W2 * h + b2)
  # y_hat = f(z)
  

  func1   <-  NULL
  
  # --- Funções de ativação de saída f(z) ---
  #___________________________________________________________________________________________________
  if(NET[["outF"]] == "linear")  func1  <-  quote(z) 
  
  if(NET[["outF"]] == "sigmoid") func1 <- bquote({
                                                    ifelse(z >= 0,
                                                           1 / (1 + exp(-z)),
                                                           exp(z) / (1 + exp(z)))
                                                  })
  #___________________________________________________________________________________________________

  
    # calculando a matriz de valores z = w2*ψ(a) + b2 
    z   <-  cbind(h, 1)  %*%  (NET[["weights"]][[2]])
  
    # armazenando z = w2*ψ(a) + b2 
    NET[["InputF"]]  <-  z 
  
    # calculando a matriz de valores f(z) = y_hat 
    y                  <-  eval(func1) 
    NET[["trainout"]]  <-  y 
    
    return(NET)

}
