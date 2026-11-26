# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .buildtree
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
# Purpose: Recursive binary tree construction for the
#          No-U-Turn Sampler (NUTS).
#
# Contribution: Adapted and extended from the original
#               BLNN implementation to accommodate
#               diagonal and dense inverse mass matrices,
#               numerical divergence monitoring,
#               acceptance statistics, and numerical
#               safeguards within the BLWNN framework.
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


# --- [1] Constrói recursivamente a trajetória (árvore) do NUTS.
#         - A cada nível j, dobra o nº de passos leapfrog.
#         - Para ao encontrar U-turn ou divergência.
.buildtree <- function(theta,                     # posição atual (parâmetros no espaço transformado)
                       r,                         # momento atual
                       logu,                      # limite do slice sampling (em log-space)
                       v,                         # direção do movimento (+1 direita, -1 esquerda)
                       j,                         # profundidade da árvore (0 = folha)
                       eps,                       # tamanho do passo leapfrog
                       H0,                        # energia inicial (Hamiltoniana)
                       fn,                        # função de log-probabilidade
                       gr,                        # gradiente da log-probabilidade
                       delta.max = 1000,          # tolerância para divergências
                       info      = environment(), # armazena estatísticas de execução
                       M_inv)                     # matriz inversa da massa
{
  # [1.1] CASO BASE: j = 0 → folha da árvore → executa um único passo leapfrog
  if(j == 0)
    {
    # [1.1.1] Meio passo no momento (r)
    r            <-  r - (v * eps / 2) * gr(theta)
    # [1.1.2] Passo completo na posição (theta)
    theta        <-  theta + (v * eps) * .apply.Minv(M_inv, r)
    # [1.1.3] Meio passo final no momento (finaliza leapfrog)
    r            <-  r - (v * eps / 2) * gr(theta)
    
    # [1.1.4] Calcula energia no novo ponto
    H            <- .calculate.H(theta = theta, r = r, fn = fn, M_inv = M_inv)
    
    # [1.1.5] Critério do slice sampling: 
    n            <- as.integer(logu <= -H)     # ponto aceito? (1 se logu <= -H, 0 caso contrário)
    
    # [1.1.6] Critério de divergência: compara salto de energia com delta.max
    #         trajetória válida? (não ultrapassou limite de energia)
    #         verifica salto de energia |H - H0| maior do que delta.max
    # [1.1.6] Critério de divergência
    deltaH <- H - H0
    if(!is.finite(H) || abs(deltaH) > delta.max) {
      info$divergent <- 1
      s              <- 0
      alpha          <- 0                    # não contabiliza taxa de aceitação
      nalpha         <- 0                    # não contribui para Dual Averaging
    } else {
      s              <- 1
      # [1.1.7] Calcula taxa de aceitação (log e normal)
      logalpha       <- H0 - H
      alpha          <- min(1, exp(logalpha))
      nalpha         <- 1                     # folha válida contribui
      info$n.calls   <- info$n.calls + 1
    }
  
    #deltaH       <- H - H0
    #if(!is.finite(H) || abs(deltaH) > delta.max) 
    #{       info$divergent <- 1   # registre divergência para diagnóstico
    #        s              <- 0
    #} else {s              <- 1
    #        }
    
    # [1.1.7] Calcula taxa de aceitação (log e normal)
   # logalpha      <-  H0 - H
    #alpha         <-  min(1, exp(logalpha))
    #info$n.calls  <-  info$n.calls + 1

    # Garantir finitos
    alpha    <- as.numeric(ifelse(is.finite(alpha), alpha, 0))
    nalpha   <- as.numeric(ifelse(is.finite(nalpha), nalpha, 0))
    n        <- as.numeric(ifelse(is.finite(n), n, 0))
    s        <- as.numeric(ifelse(is.finite(s), s, 0))
    
    
    # [1.1.8] Retorna extremos e candidato (iguais no caso base)
    return(list(theta.minus  = theta,
                theta.plus   = theta,
                theta.prime  = theta,
                r.minus      = r,
                r.plus       = r,
                s            = s,
                n            = n,
                alpha        = alpha,
                nalpha       = nalpha))    # <- usa a variável ajustada
    } else {
      
      
    # [1.2] CASO RECURSIVO: j > 0 → construir subárvore
    # [1.2.1] Constrói a primeira metade da trajetória (subárvore do lado v)
    xx            <-  .buildtree(theta = theta,
                                 r     = r,
                                 logu  = logu,
                                 v     = v,
                                 j     = j - 1,
                                 eps   = eps,
                                 H0    = H0,
                                 fn    = fn,
                                 gr    = gr,
                                 info  = info,
                                 M_inv = M_inv)
    
    # [1.2.2] Guarda extremos e estatísticas da primeira metade
    theta.minus   <- xx$theta.minus
    theta.plus    <- xx$theta.plus
    theta.prime   <- xx$theta.prime
    r.minus       <- xx$r.minus
    r.plus        <- xx$r.plus
    alpha         <- xx$alpha
    nalpha        <- xx$nalpha
    s             <- xx$s
    nprime        <- xx$n
    
    if(!is.finite(s)) s <- 0
    
    
    # [1.2.3] Se a primeira metade ainda é válida, constrói a segunda metade
    if(s == 1){
      if(v == -1){
                    # [1.2.4] Expande para a esquerda
                    yy  <- .buildtree( theta = theta.minus,
                                       r     = r.minus,
                                       logu  = logu,
                                       v     = v,
                                       j     = j - 1,
                                       eps   = eps,
                                       H0    = H0,
                                       fn    = fn,
                                       gr    = gr, 
                                       info  = info,
                                       M_inv = M_inv)
                    
                    # [1.2.5] Atualiza extremos à esquerda
                    theta.minus  <- yy$theta.minus
                    r.minus      <- yy$r.minus
                  }else {
                            # [1.2.6] Expande para a direita
                            yy   <- .buildtree( theta = theta.plus,
                                                r     = r.plus,
                                                logu  = logu,
                                                v     = v,
                                                j     = j - 1,
                                                eps   = eps,
                                                H0    = H0,
                                                fn    = fn,
                                                gr    = gr, 
                                                info  = info,
                                                M_inv = M_inv)
                            
                            # [1.2.7] Atualiza extremos à direita
                            theta.plus  <-  yy$theta.plus
                            r.plus      <-  yy$r.plus
                  }
      
      
      # [1.2.8] Soma nº de pontos aceitos
      nprime       <-  yy$n + xx$n
      if(!is.finite(nprime)) nprime  <-  0

      
      # [1.2.9] Escolhe candidato aleatoriamente na trajetória
      if(nprime > 0)
        if(runif(1) <= yy$n / nprime) theta.prime  <-  yy$theta.prime
      
      # [1.2.10] Soma taxas de aceitação
      alpha        <-  sum(c(xx$alpha, yy$alpha), na.rm = TRUE)
      nalpha       <-  sum(c(xx$nalpha, yy$nalpha), na.rm = TRUE)
          
          
      # [1.2.11] Testa U-turn e decide se continua expandindo
      b            <- .test.nuts( theta.plus  = theta.plus,
                                  theta.minus = theta.minus,
                                  r.plus      = r.plus,
                                  r.minus     = r.minus,
                                  M_inv       = M_inv)
      s            <-  yy$s * b
      
      if(info$divergent == 1 || yy$s == 0) info$divergent <- 1
    }
    
    
    # Garantir finitos antes de retornar
    alpha  <- as.numeric(ifelse(is.finite(alpha), alpha, 0))
    nalpha <- as.numeric(ifelse(is.finite(nalpha), nalpha, 0))
    nprime <- as.numeric(ifelse(is.finite(nprime), nprime, 0))
    s      <- as.numeric(ifelse(is.finite(s), s, 0))
    
    
    
    
      # [1.2.12] Retorna extremos e estatísticas da árvore
      return(list(theta.minus = theta.minus,
                  theta.plus  = theta.plus,
                  theta.prime = theta.prime,
                  r.minus     = r.minus,
                  r.plus      = r.plus,
                  s           = s,
                  n           = nprime,
                  alpha       = alpha,
                  nalpha      = nalpha))
    }
  }
