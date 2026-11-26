# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: BLWNN_Predict
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
# Purpose: Point-estimate-based and posterior-based
#          prediction using Bayesian wavelet neural
#          networks.
#
# Contribution: Adapted from the original BLNN
#               implementation to accommodate the
#               extended parameter structure and
#               prediction procedures of BLWNN.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

BLWNN_Predict   <-  function(Network, x, y = NULL, fit = NULL) 
{
  if(is.null(fit) == TRUE) 
  {
      # Alimentar a rede dada a matriz de covariáveis x. 
      forwarded  <-  .ffwd(Network, x)
      
      # Função de ativação da camada de saída 
      if(Network$outF == "softmax") {return(max.col(forwarded$trainout))
        
      } else{
              if(is.null(y) == TRUE) {return(forwarded$trainout)
                
              } else{ # Calcular o erro da rede 
                      # retorna E_D e E_theta separados 
                      Err    <-  .NetErr(y, Network, x, sep = TRUE) 
                      
                      # y - y_hat == diff
                      diff   <-  (y - forwarded$trainout)
                      
                      return(list(Errors           =  Err,
                                  Difference       =  diff,
                                  Predicted_Values =  forwarded$trainout))
                    }
      }
    
  } else{
          fil.net     <-  function(w){  nettemp   <-  BLNN:::.UpNetWts(w,Network)
                                        out       <-  BLWNN_Predict(nettemp,x)
                                        return(out)
                                      }
    
    
    ### Coletando todas as amostras aceitas do objeto de ajuste
    d1      <- NULL
    chains  <- length(fit$sampler_params)
    
    for(k in 1:chains){        m   <- fit$samples[,k,]
      
      if(fit$algorithm=="HMC"){m1  <- as.data.frame(cbind(m, acc <- fit$sampler_params[[k]][,5]))
      }else {                  m1  <- as.data.frame(cbind(m, acc <- fit$sampler_params[[k]][,7]))}
      
    
    
      if(chains == 1){d1   <- m1 %>%  dplyr::filter(acc == 1) %>% dplyr::select(-Er__,-acc)
        }else {       d1   <- rbind(d1, m1 %>% dplyr::filter(acc == 1) %>% dplyr::select(-Er__,-acc))}}
    
    
    pred   <-   apply(d1, 1, fil.net)
    
    if(Network$outF == "softmax")
    {
      return(apply(pred, 
                   1, 
                   function(x)
                   {as.numeric(names(table(x))[table(x)==max(table(x))])}))
    }else{
      # Calculando a média das colunas 
      return(colMeans(t(pred)))
    }
    
  }
}
