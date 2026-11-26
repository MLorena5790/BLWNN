# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .rotate_space
#
# Original implementation: BLNN developers
# Source: https://github.com/BLNNdevs/BLNN
#
# Adaptations and modifications:
# Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
# (modifications)
#
# Developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Linear parameter-space transformation for
#          Hamiltonian Monte Carlo and NUTS sampling.
#
# Contribution: Adapted from the original BLNN
#               implementation to support parameter-space
#               transformations within the BLWNN framework.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================
# [5.1] Função p/ rotacionar o espaço de parâmetros (usando M) p/ melhorar a eficiência do HMC. 
    #       Leva os parâmetros init(q) para x.cur(x) (espaço transformado). 
    #       Aplicar mudança de variáveis: q = A·x  ⇒  x = A⁻¹·q
    #       Essa rotação ajuda a padronizar a geometria do espaço, acelerando o leapfrog.
    .rotate_space  <-  function (fn, gr, M, theta.cur)
    {
      # [5.1.1] Caso 1: M é uma matriz completa (massa densa, com correlações) 
      #                 Transformação linear do espaço: x = L⁻¹·q, com M = L·Lᵗ
      if(is.matrix(M)) 
      {
        chd      <-  t(chol(M))                                      # L = t(chol(M)) matriz triangular inferior tq M = L × Lᵗ = diag(M)
        chd.inv  <-  solve(chd)                                      # L⁻¹ — usada para mudar de coordenadas q → x
        fn2      <-  function(x) fn(chd %*% x)                       # f̃(x) = f(L·x) fn2 = fn(q = L.x)
        gr2      <-  function(x) as.vector(t(chd) %*% gr(chd %*% x)) # ∇f̃(x) = ∇f(q) . L, com q = L·x
        x.cur    <-  as.numeric(chd.inv %*% theta.cur)                   # q' = x = L⁻¹·q  — coordenadas transformadas
        
        # [5.1.2] Caso 2: M é um vetor (massa diagonal: M = diag(m₁, ..., m_n))
        #                 Transformação de escala: x = D⁻¹·q, com D = diag(√M)
        #                 M como matriz de massa diagonal ⇒ sem correlações entre parâmetros
      }else if(is.vector(M))
      {
        chd    <-  sqrt(M)                                  # D = diag(√m₁, ..., √m_n)
        fn2    <-  function(x) fn(chd * x)                  # f̃(x) = f(D·x)
        gr2    <-  function(x) chd * gr(chd * x)            # ∇f̃(x) = ∇f(q) · D
        x.cur  <-  (1 / chd) * theta.cur                    # x = D⁻¹·q
      }
      
      # [5.1.3] Verificar se M é matriz ou vetor 
      else{stop("Mass matrix must be vector or matrix")}
      
      # [5.1.4] Retorna funções e parâmetros transformados no novo espaço
      return(list(gr2   = gr2,   # Gradiente transformado: ∇f̃(x)
                  fn2   = fn2,   # Função objetivo transformada: f̃(x)
                  x.cur = x.cur, # Valor inicial transformado: x = A⁻¹·q
                  chd   = chd))  # Matriz de transformação: A (L ou D)
    }
    
