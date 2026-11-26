# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .NetErr
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
# Purpose: Evaluation of the Bayesian objective function,
#          combining data error and parameter-specific
#          regularization terms.
#
# Contribution: Adapted from the original BLNN implementation
#               to accommodate the Bayesian wavelet neural
#               network architecture and its additional
#               parameter groups.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

# .NetErr função p/ calcular M(w) = β*E_D + α*E_W
.NetErr  <-  function(Actual,           # saidas/respostas verdadeiras 
                      NET,              # objeto BLWNN 
                      sep = FALSE, ...) # Retorno do erro total ou separado por grupo
{
  
  
  # --- [1] Propagação direta (feedforward)
  fwded  <-  .ffwd(NET)          
  yhat   <-  fwded[["trainout"]] # y_hat
  
  
  # --- [2] Cálculo de E_D: Erros dos dados ---
  Err <- if(fwded[["costF"]] == "crossEntropy") 
            {
              # Estabilização numérica para evitar log(0)
              eps  <- .Machine$double.eps
              yhat <- pmin(pmax(yhat, eps), 1 - eps)
    
              -sum(Actual * log(yhat) + (1 - Actual) * log(1 - yhat))
    
            }else{ #mean((Actual - yhat)^2)
              0.5 * sum((Actual - yhat)^2)} # Para SSE
  
  
  # --- [3] Cálculo de M(w) = β·E_D(w) + α·E_W(w) via .ErrBayes ---
  if(sep == TRUE)
  {
    BRS  <-  .ErrBayes(fwded, Err, sep = TRUE)   

    return(list(Total    = BRS$edata + BRS$EP_all, # M(w) = β.E_D(w) + α.E_W(w) → tds parâmetros (treinamento)
                Bayes    = BRS$EP_all,             # α.E_W(w) total: W1, b1, W2, b2, t_tilde, log(r)
                EVEP_all = BRS$EVEP_all,           # E_W(w) puro por grupo W1, b1, W2, b2, t_tilde, log(r)
                Data     = Err))                   # E_D(w): erro de dados puro
    
  } else {
            FinalErr  <-  .ErrBayes(fwded, Err, sep = FALSE)    # Retorna apenas o erro total M(w) = β.E_D(w) + α.E_W(w)
            return(FinalErr)
          }
}
