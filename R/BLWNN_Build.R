# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: BLWNN_Build
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
# Purpose: Construction and initialization of Bayesian
#          wavelet neural networks.
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



BLWNN_Build   <-  function(norm_data,                  
                           nout,
                           hlayer_size       = 5,
                           waveF             = c("sig_dev2", "mexicanhat"), 
                           costF             = c("MSE","crossEntropy"),
                           outF              = c("linear", "sigmoid"),
                           hp.Err            = FALSE, # FALSE = sem regularização
                           hp.W1             = FALSE,
                           hp.W2             = FALSE,
                           hp.B1             = FALSE,
                           hp.B2             = FALSE,
                           hp.t.tilde        = FALSE, 
                           hp.log.r          = FALSE)    

  {    

          nn <- NULL 
          
          ### --- Verificação de entrada 'norm_data' --- ###
          if (!is.matrix(norm_data) || !is.numeric(norm_data) || nrow(norm_data) < 1 || ncol(norm_data) < 1)
            stop("Input 'norm_data' must be a numeric matrix of covariates.")
          nn$ncov  <- ncol(norm_data)
          
          
          ### --- Verificação de 'nout' --- ###
          if (length(nout) != 1 || !is.numeric(nout) || nout <= 0 || nout != as.integer(nout)) 
            stop("Number of output variables must be a single positive integer")
          nn$nout <- as.integer(nout)
  
          
          ### --- Verificação de 'hlayer_size' --- ###
          if (length(hlayer_size) != 1 || !is.numeric(hlayer_size) || hlayer_size <= 0 || 
              hlayer_size != as.integer(hlayer_size))
            stop("Hidden layer size must be a single positive integer greater than zero")
          nn$hidden_size <- as.integer(hlayer_size)
          
          ### --- Verificação de 'waveF', 'costF' e 'outF' --- ###
          waveF    <- match.arg(waveF); costF    <- match.arg(costF); outF     <- match.arg(outF)
          nn$waveF <- waveF; nn$costF <- costF; nn$outF  <- outF
          
          
          ### --- Verificação dos hiperparâmetros --- ###
          hparams_all <- list(hp.Err, hp.W1, hp.B1, hp.W2, hp.B2, hp.t.tilde , hp.log.r)
          if(any(sapply(hparams_all, function(x) is.numeric(x) && any(x < 0)))) 
            stop("Hyperparameters must be non-negative when specified.")
          
          
          hparams <- list(hp.Err, hp.B1, hp.W2, hp.B2, hp.t.tilde , hp.log.r)
          if(any(sapply(hparams, function(x) !identical(x, FALSE) && length(x) != 1))) 
            stop("All hyperparameter values (except hp.W1) must be scalars when specified.")
          
          if(!identical(hp.W1, FALSE) && !(length(hp.W1) == 1 || length(hp.W1) == ncol(norm_data))) 
            stop("Hyperparameter 'hp.W1' must be of length 1 or match the number of covariates.")
          
          
          
          ### --- Armazenar hiperparâmetros --- ###
          nn$scale.error   <- hp.Err
          nn$scale.weights <- list(hp.W1      = hp.W1, 
                                   hp.B1      = hp.B1, 
                                   hp.W2      = hp.W2, 
                                   hp.B2      = hp.B2, 
                                   hp.t.tilde = hp.t.tilde , 
                                   hp.log.r   = hp.log.r)
          
          

          ### --- Inicialização pesos/viéses (NÃO são priors bayesianos) --- ###
          # Inicialização tipo Xavier:
          # W1_0, B1_0 ~ Normal(0, 1/p), p = ncov. 
          nn$weights      <-  list() 
          
          scale_w1        <- 1 / sqrt(ncol(norm_data))
          nn$weights[[1]] <- matrix(rnorm((ncol(norm_data) + 1) * hlayer_size) * scale_w1, 
                                    nrow = ncol(norm_data) + 1, 
                                    ncol = hlayer_size) # W1, B1
          
          
          # W2_0, B2_0 ~ Normal(0, 1/H), H = n. wavelon
          scale_w2        <- 1 / sqrt(hlayer_size)
          nn$weights[[2]] <- matrix(rnorm((hlayer_size + 1) * nout) * scale_w2, 
                                    nrow = hlayer_size + 1, 
                                    ncol = nout)       # W2, B2
        
          
          ### --- Armazenando dados normalizados --- ###
          nn$normData     <- norm_data
        
          
          ### --- Inicialização parâmetros wavelets / reparametrização não centrada --- ###
          # 1) Inicializamos log(r) diretamente no espaço real
          # log(r) ~ N(0, 0.25^2)  →  r ∈ [0.61, 1.65] (≈95%), com r = 1/s e dados normalizados 
          mu_log_r       <- 0     # (NOVO!)
          sd_log_r       <- 0.25  # (NOVO!)
          log.r          <- rnorm(hlayer_size, mean = mu_log_r, sd = sd_log_r) # (NOVO!)
          #log.r          <- pmin(pmax(log.r, -5), 5)
        
          # 2) Amostramos uma variável auxiliar independente
          #    t_tilde ~ Normal(0, 1)
          t_tilde        <- rnorm(hlayer_size, mean = 0, sd = 1)
            
          # 3) Definimos a translação como escala × variável auxiliar
          # t_tilde é o parâmetro de translação reparametrizado
          # r = exp(log.r), com r = 1/s
          # A translação original é recuperada por t = t_tilde/r
            
          nn$weights[[3]] <- matrix(t_tilde, ncol = 1)  # t_tilde 
          nn$weights[[4]] <- matrix(log.r, ncol = 1)    # log(r), com r = 1/s (NOVO!)

          
          ### --- Preparação de estruturas para propagação forward --- ###
          # waveInput:entrada da wavelet, u.r - t_tilde
          # waveOut:   saída da wavelet, ψ(u.r - t_tilde)
          # InputF:    entrada da função de ativação de saída, z = W2.h + b2
          # trainout:  saída final da rede, y = f(z)
          
          nn$waveInput    <-  list()
          nn$waveOut      <-  list()
          nn$InputF       <-  list()
          nn$trainout     <-  list()
          
          
          class(nn)       <-  "BLWNN"
          return(nn) 
  
  }

