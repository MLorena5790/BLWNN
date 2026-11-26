# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .update_control
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
# Purpose: Configuration and management of control
#          parameters for HMC and NUTS sampling.
#
# Contribution: Adapted and extended from the original
#               BLNN implementation to incorporate
#               metric configuration, warmup adaptation
#               windows, and parameter block information
#               within the BLWNN framework.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

# --- [1] Função p/ configurar parâmetros de controle do HMC/NUTS (default)  ---
  #         Define constantes físicas do sistema hamiltoniano
  .update_control <- function(control)
  { 
    # [1.1] Valores padrão para os parâmetros de controle do HMC/NUTS
    default   <-   list( #both NUTS and HMC
      adapt_delta   = 0.65, # Taxa (%) de aceitação (HMC -> 0.65)
      momentum.mass = 1,    # Variância do momento (massa)
      stepsize      = NULL, # Tamanho de passo (ε)
      useDA         = TRUE, # Dual Averaging (ajustar ε de forma adaptativa)
      gamma         = 0.05, # Taxa de aprendizado DA (Hoffman & Gelman, 2014)
      t0            = 10,   # Parâmetro de suavização DA (Hoffman & Gelman, 2014)
      kappa         = 0.75, # Decaimento DA (Hoffman & Gelman, 2014)
      metric        = NULL, # Métrica da massa 
      adapt_mass    = FALSE,# Adaptação da métrica
      w1            = 75,   # Janela 1 de warmup (para adaptação da massa)
      w2            = 50,   # Janela 2
      w3            = 25,   # Janela 3
      max_treedepth = 10,   # Profundidade máxima da árvore NUTS — NUTS apenas
      Lambda        = 0.25, # Comprimento da trajetória (λ = ε × L) — HMC apenas
      param_blocks  = NULL) # índices dos blocos de parâmetros (opcional)
    
    
    # Merge inicial
    if(is.null(control)) {new <- default
    } else {              new <- modifyList(default, control)}
    
    # Se metric foi fornecida → ignorar momentum.mass
    if(!is.null(new$metric)) {new$momentum.mass <- NULL
    new$adapt_mass    <- FALSE}
    
    # [1.4] Lista final c/ parâmetros de controle HMC/NUTS
    return(new)
  }
  
