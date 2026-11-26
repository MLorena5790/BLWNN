# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .test.nuts
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
# Purpose: Evaluation of the U-turn stopping criterion
#          for the No-U-Turn Sampler (NUTS).
#
# Contribution: Adapted and modified from the original
#               BLNN implementation to incorporate the
#               inverse mass matrix into the U-turn
#               stopping criterion, supporting diagonal
#               and dense mass matrices.
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



## Testa se ocorreu um "U-turn" (retorno em U) em um ramo da árvore binária
## construída pela função `.buildtree`.
## Um "U-turn" indica que a trajetória Hamiltoniana voltou sobre si mesma,
## sinalizando que a expansão da árvore deve ser interrompida para evitar
## amostras redundantes e garantir a eficiência da amostragem.
##
## Parâmetros:
## - theta.plus: vetor posição no extremo direito da trajetória (parâmetros)
## - theta.minus: vetor posição no extremo esquerdo da trajetória
## - r.plus: vetor momento correspondente ao extremo direito
## - r.minus: vetor momento correspondente ao extremo esquerdo
##
## Retorna:
## - TRUE (1) se **não** houve U-turn (trajetória ainda válida para expansão)
## - FALSE (0) se houve U-turn (deve parar a expansão da árvore)

.test.nuts <- function(theta.plus, theta.minus, r.plus, r.minus, M_inv) 
  {
    # Diferença entre extremos da trajetória
    theta.temp <- theta.plus - theta.minus
    
    # Verifica se a direção do momento em cada extremo está alinhada com o vetor da trajetória
    # Usamos o produto interno (crossprod) para testar o ângulo entre theta.temp e r.minus/r.plus
    # Se ambos os produtos internos forem >= 0, não houve U-turn
    left  <- as.numeric(crossprod(theta.temp, .apply.Minv(M_inv, r.minus)))
    right <- as.numeric(crossprod(theta.temp, .apply.Minv(M_inv, r.plus)))
    
    # Retorna 1 (TRUE) ou 0 (FALSE)
    return((left >= 0) && (right >= 0))
  }
