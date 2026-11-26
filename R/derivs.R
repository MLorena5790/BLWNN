# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .derivs
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
# Purpose: Analytical gradient computation through
#          backpropagation in Bayesian wavelet neural
#          networks.
#
# Contribution: Adapted and substantially extended from
#               the original BLNN implementation to
#               incorporate wavelet activation derivatives
#               and analytical gradients with respect to
#               reparameterized wavelet translation and
#               dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.derivs  <-  function(
                      Actual,   # saidas verdadeiras y
                      NET       # objeto .BLWNN_Build 
                      )
  
{
  
  # --- [1] Propagação direta (feedforward) 
  network    <-    .ffwd(NET) 
  
  
  # --- [1.1] Validar a existência de estruturas primordiais
  stopifnot(!is.null(network$wavePre))
  stopifnot(!is.null(network$waveInput))
  stopifnot(!is.null(network$waveOut))
  
  
  # --- [1.2] Gradientes válidos apenas para: (linear + MSE) ou (sigmoid + crossEntropy)
  stopifnot( (network$outF == "linear"  && network$costF == "MSE") ||
             (network$outF == "sigmoid" && network$costF == "crossEntropy"))
  
  
  # --- [2] Derivadas das funções wavelets ψ'(a)
  #___________________________________________________________________________________________________
  # Limites para 'a' em sig_dev2:
  # - Superior: -log(.Machine$double.eps) ≈ 36 (evita exp(a) = Inf)
  # - Inferior: -log(1/.Machine$double.xmin - 1) ≈ -708 (evita exp(a) = 0)
  if(network[["waveF"]] == "sig_dev2")   DactF  <- bquote({
                                                        a_tr <-  pmin(a, -log(.Machine$double.eps))        
                                                        a_tr <-  pmax(a_tr, -log(1/.Machine$double.xmin - 1)) 
                                                        sig  <- 1 / (1 + exp(-a_tr))                       
                                                        -6 * sig**4 + 12 * sig**3 - 7 * sig**2 + sig
                                                      })

  if(network[["waveF"]] == "mexicanhat") DactF  <- bquote({ a_tr     <- pmin(pmax(a, -10), 10)               # Limites mais conservadores
                                                            exp_part <- exp(-a_tr^2/2)                       # Computação mais estável
                                                            exp_part <- pmax(exp_part, .Machine$double.xmin) # Evitar overflow/underflow
                                                            (a_tr^3 - 3*a_tr) * exp_part
                                                          })
  #___________________________________________________________________________________________________
  
  
  # --- [3] Obter informações relevantes da Propagação direta ---
  u         <-  network[["wavePre"]]      # u = w1*x + b1
  a         <-  network[["waveInput"]]    # a = (u.r) - t_tilde 
  yhat      <-  network[["trainout"]]     # ŷ 
  
  
  pa        <-  network[["waveOut"]]        # h = ψ(a) 
  norm_data <-  network[["normData"]]       # entradas normalizadas
  r         <-  exp(NET[["weights"]][[4]])  # r = exp(log(r)) = 1/s
  
  if(any(!is.finite(r)) || any(r < 1e-8))
    stop("Invalid dilation encountered")
  
  
  # --- [4] Erro na saída (ŷ - y) ---
  outError    <-  (yhat - Actual)        
  
  
  
  # --- [5] Derivadas da camada de saída (W2 e b2) ---
  Gw2         <-  t(pa) %*% outError  # ∂C/∂W2 
  Gb2         <-  colSums(outError)   # ∂C/∂b2 
  

  # --- [6] Inicialização das listas para gradientes e erros de camada ---
  # RCweights   -> (∂C/∂W1, ∂C/∂b1, ∂C/∂W2, ∂C/∂b2, ∂C/∂t_tilde, ∂C/∂log(r)) 
  # layerErrors -> erros propagados
  
  RCweights          <-  list()
  layerErrors        <-  list()

  
  # --- [7] Armazenar δ₂ = (ŷ - y) como erro da camada 2(saída) --
  layerErrors[[2]]   <- outError # (y_hat - y) 
  
  
  # --- [8] Retropropagação para camada oculta (wavelons) ---
  # erro da camada 1 (oculta): δ₁ = (δ₂ %*% t(W2)) * ψ'(a)
  # eval(DactF) = ψ'(a) 
  layerErrors[[1]]  <-  (layerErrors[[2]] %*% 
                        t(network[["weights"]][[2]][-nrow(network[["weights"]][[2]])])) * 
                        eval(DactF)
  
  # ∂C/∂t_tilde = - δ₁
  # r = exp(log(r)) = 1/s
  RCweights[[3]]   <-  sweep(matrix(colSums(layerErrors[[1]]), ncol = 1), 
                             1, 
                             -1, 
                             FUN="*")
  
  # ∂C/∂log(r) = δ₁ * u * r
  # r = exp(log(r)) = 1/s  

  RCweights[[4]]   <- matrix(colSums(sweep((layerErrors[[1]] * u),
                                              2, 
                                              r, 
                                              FUN="*")), ncol = 1) 
  
  
  
  # ∂C/∂W1 = t(x) %*% (δ₁ * r)
  # r = exp(log(r)) = 1/s
  RCweights[[1]]   <- t(norm_data) %*%  sweep(layerErrors[[1]], 
                                              2, 
                                              r, 
                                              FUN="*") 
  
  # ∂C/∂b1 =  δ₁ * r
  # r = exp(log(r)) = 1/s
  Gb1              <-  colSums(sweep(layerErrors[[1]], 
                                     2, 
                                     r, 
                                     FUN="*"))     
  
  # --- [9] Agrupamento: ∂C/∂W1 e ∂C/∂b1 --- 
  RCweights[[1]] <-  rbind(RCweights[[1]], Gb1)  
  
  
  # --- [10] Agrupamento: ∂C/∂W2 e ∂C/∂b2 ---  
  RCweights[[2]] <-  rbind(Gw2, Gb2)      
  
  
  # --- [11] Cálculo de ∇M(w) = β * ∇E_D(w) + α * ∇E_W(w)
  #                     ∂M(θ)/∂θ = β * ∂E_D/∂θ + α * ∂E_θ/∂θ
  FinalGrad      <-   .DBayes(network, RCweights)  
  
  # Sanear NAs / Infs sem alterar a estrutura
  FinalGrad      <-   lapply(FinalGrad, function(g) {g[!is.finite(g)] <- 0
                                                     g})
  return(FinalGrad)
  
  }
