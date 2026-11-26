# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .normalization_test
# Author: Márcia Lorena Alves dos Santos
# Copyright (C) 2026 Márcia Lorena Alves dos Santos
#
# Original function developed as part of the doctoral thesis:
# "Wavelets e aprendizagem bayesiana de redes neurais"
#
# Purpose: Preprocessing and normalization of new data
#          using normalization parameters stored during
#          the training stage.
#
# Contribution: Originally developed for BLWNN.
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

     
.normalization_test <-  function(new_data, object)
    {
          # --- Verificação de inputs ---
          stopifnot(is.data.frame(new_data),
                    all(sapply(new_data, function(x) inherits(x, c("numeric", "integer", "factor")))),
                    identical(names(object$norm_input), names(new_data)),
                    is.numeric(object$params$truncate),
                    object$params$truncate > 0 
                    )


          # --- Identificação das variáveis --- 
          rank_encoded   <- names(object$params$rank_params)
          mad_scaled     <- names(object$params$mad_params)
          not_normalized <- setdiff(names(new_data), c(rank_encoded, mad_scaled))
       
          
          ## --- Verificação de Variáveis Binárias ---
          if(length(not_normalized) > 0) 
          stopifnot(all(sapply(new_data[not_normalized], function(x) !any(is.na(x)) && all(x %in% c(0, 1)))))
          
          
          # --- Tratar categorias inéditas no teste ---
          if(length(rank_encoded) > 0) 
          {
                  train_levels <- lapply(object$params$rank_params, `[[`, "levels")
                  test_levels  <- lapply(new_data[rank_encoded], levels)
                  new_levels   <- Map(setdiff, test_levels, train_levels)
                  
                  # Substituir por NA os valores não vistos no treino
                  vars_with_new <- names(new_levels)[lengths(new_levels) > 0]
                  invisible(lapply(vars_with_new, 
                                   function(var) 
                                     { new_data[[var]][new_data[[var]] %in% new_levels[[var]]] <<- NA}))
          }

          
          truncate <- object$params$truncate
          
          # --- Rank enconding com N(0,1) ---  (parâmetros treino)
          expr_ord <- bquote({
                              varname  <- noquote(name)
                              x        <- new_data[[(varname)]]
                              a_rank   <- object$params$rank_params[[(varname)]]$a_rank
                              n        <- object$params$rank_params[[(varname)]]$n
                              rank_br  <- rank(x, ties.method = "average", na.last = "keep") 
                              rank_per <- (rank_br - a_rank) / (n + 1 - 2*a_rank) 
                              rank_per <- pmin(pmax(rank_per, 1e-6), 1 - 1e-6)
                              z_score  <- qnorm(rank_per)               
                              z_trunc  <- pmin(pmax(z_score, -truncate), truncate)
                              ifelse(is.na(z_trunc),
                                     object$params$rank_params[[varname]]$rank_mean,
                                     z_trunc)
                            })
          

          # ---  Robust Scaling com MAD (cte 1.4826) --- (parâmetros treino)
          expr_num <- bquote({
                                varname <- noquote(name)
                                x       <- new_data[[(varname)]]
                                x_num   <- as.numeric(as.character(x))
                                med     <- object$params$mad_params[[(varname)]]$median
                                mad_val <- object$params$mad_params[[(varname)]]$mad
                                z_score <- (x_num - med) / mad_val 
                                pmin(pmax(z_score, -truncate), truncate)
                              })

          
          #  --- Aplicação das transformações ---
          if(length(rank_encoded) > 0)   new_data[rank_encoded]   <- lapply(rank_encoded, function(name) eval(expr_ord))
          if(length(mad_scaled) > 0)     new_data[mad_scaled]     <- lapply(mad_scaled, function(name) eval(expr_num))
          if(length(not_normalized) > 0) new_data[not_normalized] <- lapply(new_data[not_normalized], function(x) as.numeric(as.character(x))) 
  

          
          return(data.matrix(new_data))
}

