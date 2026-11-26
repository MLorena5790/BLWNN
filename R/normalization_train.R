# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .normalization_train
# Author: Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
#
# Original function developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Training data preprocessing and normalization.
# Contribution: Originally developed for BLWNN.
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.normalization_train <-  function(input_data, 
                                  equidist_categ = NULL, 
                                  truncate = 2)
  {
      # --- Verificação de inputs ---
      stopifnot(is.data.frame(input_data),
                is.null(equidist_categ) || is.character(equidist_categ),
                is.numeric(truncate),
                truncate > 0, 
                all(sapply(input_data, function(x) inherits(x, c("numeric", "integer", "factor")))))

      is_ordered <- function(x) inherits(x, "ordered")
      

      # --- Identificação das variáveis --- 
      ordered_vars   <- names(input_data)[sapply(input_data, is_ordered)]
      factor_vars    <- names(input_data)[sapply(input_data, is.factor)]
      
      
      rank_encoded   <- if(is.null(equidist_categ)) ordered_vars else setdiff(ordered_vars, equidist_categ)
      not_normalized <- setdiff(factor_vars, ordered_vars)
      mad_scaled     <- setdiff(names(input_data), c(rank_encoded, not_normalized))
      
      ## --- Verificação de Variáveis Binárias ---
      if(length(not_normalized) > 0) 
        stopifnot(all(sapply(input_data[not_normalized], function(x) !any(is.na(x)) && all(x %in% c(0, 1)))))
      
     
      # --- Função de soft clipping ---
      soft_clip  <- function(z, c){c * tanh(z / c)}
      
      # --- Rank enconding com N(0,1) ---  
      expr_ord   <- bquote({
                          a_rank    <- ifelse(length(x) <= 100, 0.5, 0.001)  # Define 'a_rank' conforme n
                          rank_br   <- rank(x, ties.method = "average") 
                          rank_per  <- (rank_br - a_rank) / (length(x) + 1 - 2*a_rank) # Ajuste para (0, 1)
                          rank_per  <- pmin(pmax(rank_per, 1e-8), 1 - 1e-8)  # elimina os riscos de NaN no qnorm()
                          z_score   <- qnorm(rank_per)                       # Transformação para N(0,1)
                          #z_trunc   <- pmin(pmax(z_score, -truncate), truncate)
                          z_trunc   <- soft_clip(z_score, truncate)
                          return(list(z = z_trunc, 
                                      params = list(a_rank    = a_rank, 
                                                    n         = length(x),
                                                    levels    = levels(x), 
                                                    rank_mean = mean(z_score))))
                          
                        })
      

      # ---  Robust Scaling com MAD (cte 1.4826) --- 
      expr_num <- bquote({
                          x_num   <- as.numeric(as.character(x))
                          med     <- median(x_num)
                          mad_val <- mad(x_num, constant = 1.4826)
                          #z_score <- if (mad_val == 0) rep(0, length(x_num)) else (x_num - med) / mad_val #Protecão mad = 0
                          mad_val <- max(mad_val, 1e-6)
                          z_score <- (x_num - med) / mad_val
                          z_trunc <- soft_clip(z_score, truncate)
                          return(list(z = z_trunc, 
                                      params = list(median = med, 
                                                    mad = mad_val)))
                          })
      
      
      # ---  Objetos para guardar os parâmetros ---  
      rank_params <- mad_params <- list()
      
      #  --- Aplicação das transformações ---
      if(length(rank_encoded) > 0)   
      { 
        results                   <- lapply(input_data[rank_encoded], function(x) eval(expr_ord))
        input_data[rank_encoded]  <- lapply(results, `[[`, "z")
        rank_params[rank_encoded] <- lapply(results, `[[`, "params")
      }
      
      if(length(mad_scaled) > 0)
      {
        results                <- lapply(input_data[mad_scaled], function(x) eval(expr_num))
        input_data[mad_scaled] <- lapply(results, `[[`, "z")
        mad_params[mad_scaled] <- lapply(results, `[[`, "params")
      }

      if(length(not_normalized) > 0) 
      { input_data[not_normalized] <- lapply(input_data[not_normalized], function(x) as.numeric(as.character(x))) }
      
      
      return(list(norm_input = data.matrix(input_data), 
                  params     = list(  rank_params = rank_params, 
                                      mad_params  = mad_params, 
                                      truncate    = truncate 
                                    )))
      }
      
