# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .find.epsilon
#
# Original implementation: BLNN developers
# Source: https://github.com/BLNNdevs/BLNN
#
# Adaptations and modifications:
# Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
# (modifications and extensions)
#
# Developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Initial step-size estimation for Hamiltonian
#          Monte Carlo and NUTS sampling.
#
# Contribution: Adapted and modified from the original
#               BLNN implementation to accommodate
#               diagonal and dense inverse mass matrices
#               and numerical safeguards.
#
# Methodological reference:
# Hoffman, M. D. and Gelman, A. (2014).
# The No-U-Turn Sampler: Adaptively Setting Path Lengths
# in Hamiltonian Monte Carlo.
# Journal of Machine Learning Research, 15, 1593-1623.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

# --- [8] Função para encontrar um valor razoável de ε (stepsize) inicial p/ DA. --- 
  #         Estratégia: dobra ou divide ε até que a aceitação seja > 0.5
  #         Baseado em Hoffman & Gelman (2014, p. 1608)
  .find.epsilon <- function( theta,          # valor atual dos parâmetros: θ
                             fn,             # energia potencial: U(θ) = -log π(θ)
                             gr,             # gradiente: ∇U(θ)
                             eps     = 1,    # valor inicial de stepsize
                             verbose = TRUE,
                             M_inv)          # massa inversa (vetor: assume diagonal)
  {
    
    # [8.1] Amostragem do momentum inicial p ~ N(0, M^{-1}) 
    #       Fixo durante o ajuste de ε
    #r <- rnorm(n = length(theta), mean = 0, sd = sqrt(1/M_inv))
    
    if(is.vector(M_inv)){     r <- rnorm(length(theta), 0, sqrt(1/M_inv))
    } else {# M_inv é matriz
      cholM <- chol(solve(M_inv))  # porque M_inv = M^{-1}
      r <- as.vector(t(cholM) %*% rnorm(length(theta)))}
    
    
    # [8.2] Função auxiliar: calcula o Hamiltoniano H(θ, p) = U(θ) + K(p), 
    #                        onde U(θ) = fn(θ) e K(p) = (1/2) * pᵗ M⁻¹ p
    #.hamiltonian <-  function(theta, r, fn, M_inv) 
    #{
    #  fn(theta) + (1 / 2) * sum(M_inv * r ^ 2) 
    #}
    
    .hamiltonian <- function(theta, r, fn, M_inv)
    {
      kinetic <- if(is.vector(M_inv)){ sum(M_inv * r^2) / 2
      } else {as.numeric(t(r) %*% M_inv %*% r) / 2}
      
      fn(theta) + kinetic
    }
    
    
    # [8.3] Um passo leapfrog com ε atual
    r.new     <- r - (eps / 2) * gr(theta)           # Meio passo no momentum
    #theta.new <- theta + eps * (M_inv * r.new)      # Passo completo na posição
    if(is.vector(M_inv)){ theta.new  <- theta + eps * (M_inv *r.new)
    } else {              theta.new  <- theta + eps * as.vector(M_inv %*% r.new)}
    r.new     <- r.new - (eps / 2) * gr(theta.new)   # Meio passo final 
    
    
    # [8.4] Calcular a energia antes e depois do leapfrog
    H1        <- .hamiltonian(theta = theta, r = r, fn = fn, M_inv = M_inv)        # Fixo durante o ajuste de ε   
    H2        <- .hamiltonian(theta = theta.new, r = r.new, fn = fn, M_inv = M_inv)
    
    
    # [8.5] Qual é a probabilide de aceitação da proposta do leapfrog? 
    #       Queremos aceitação ≈ 0.5. 
    #       Se > 0.5 → a = 1, se <  0.5 → a = -1. (determina a direção do ajuste)
    accept_prob          <- exp(min(0, H1 - H2))             # evita valores > 1
    a                    <- ifelse(accept_prob > 0.5, 1, -1) # a é fixo por construção teórica
    if(!is.finite(a)) a  <- -1
    
    
    
    # [8.6] Ajustar ε até que ΔH = H2 - H1 aceitável: a*ΔH < -a*log(2)
    #       Se accept_prob > 0.5, então a = 1 e o critério vira ΔH < log(2) → Dobra eps
    #       Se accept_prob < 0.5, então a = -1 e o critério vira -ΔH < log(2) → Dividir eps
    #       Se accept_prob ≈ 0.5, critério satisfeiro a*ΔH < -a*log(2), parar!
    k  <- 1  # contador de iterações
    
    # O loop continua enquanto:
    #                         - !is.finite(H1)             : leapfrog explodiu → ε mto grande
    #                         - !is.finite(H2)             : leapfrog explodiu → ε mto grande
    #                         - a * (H2 - H1) > -a * log(2): aceitação acima ou abaixo de 0.5
    while(!is.finite(H1) | !is.finite(H2) | a * (H2 - H1) > -a * log(2))
    {
      
      # [8.7] Ajusta ε: dobra ou divide por 2
      #               - a = 1  → eps = eps*2
      #               - a = -1 → eps = eps*0.5      
      eps       <- eps * ifelse(a == 1, 2, 0.5)  
      
      
      # [8.8] Reexecutar o passo leapfrog com novo ε
      r.new     <- r - (eps / 2) * gr(theta)
      #theta.new <- theta + eps * (M_inv * r.new)   
      if(is.vector(M_inv)){ theta.new  <- theta + eps * (M_inv *r.new)
      } else {              theta.new  <- theta + eps * as.vector(M_inv %*% r.new)}
      r.new     <- r.new - (eps / 2) * gr(theta.new)
      
      
      # [8.9] Recalcular a energia
      H2        <- .hamiltonian(theta = theta.new, r = r.new, fn = fn, M_inv)
      k         <- k + 1
      
      
      # [8.10] EEvita looping infinito — limite de 50 iterações
      if(k > 50) 
      {stop("Não foi possível encontrar um epsilon razoável após 50 tentativas. 
             Verifique a especificação do modelo ou da massa.")}
    }
    
    # [8.11] Exibe o valor encontrado, se verbose = TRUE
    if(verbose)
      message(paste("Epsilon  razoável=", eps, "encontrado depois de", k, "passos"))
    
    # [8.12] Retorna ε, de forma invisível (verbose = FALSE)
    return(invisible(eps))
  }
  
