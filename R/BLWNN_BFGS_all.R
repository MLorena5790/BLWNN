# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .BLWNN_BFGS_all
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
# Purpose: Joint MAP estimation of all Bayesian wavelet
#          neural network parameters using BFGS.
#
# Contribution: Adapted and substantially extended from
#               the original BLNN implementation to
#               incorporate joint optimization of weights,
#               biases, and reparameterized wavelet
#               translation and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.BLWNN_BFGS_all  <-   function(NET,        # objeto BLWNN 
                              actual,     # respostas verdadeiras y
                              iter = 500) # Número máx. de iterações BFGS
{
  
  NETWORK   <-  NET
  ACTUAL    <-  actual
  
  # --- Vetor de pesos e vieses iniciais w₀ = c(w1, b1, w2, b2,t_tilde,log(r)) ---
  #                                      w₀ é definido em BLWNN_Build
  init.all   <-  c(t(NETWORK[["weights"]][[1]]), t(NETWORK[["weights"]][[2]]), 
                  t(NETWORK[["weights"]][[3]]), t(NETWORK[["weights"]][[4]])) # W1 + b1, W2 + b2 + t_tilde + log(r)
  
  # --- Função de M(w) = β*E_D(w₀) + α*E_W(w₀) ---
  fn1  <-  function(init.all)
                          {
                            # --- [1] Atualizar BLWNN_Build com w₀ atuais
                            network  <-  .UpNetWts(wts = init.all, NET = NETWORK)
                        
                            
                            # --- [2] Propagação direta (feedforward) com w₀ atuais
                            fwded    <-  .ffwd(network)                     
                            yhat     <-  fwded[["trainout"]] 
                            
                            # --- [3] Estabilização numérica para uso yhat no log
                            eps      <- .Machine$double.eps
                            yhat     <- pmin(pmax(yhat, eps), 1 - eps)
                            
                            
                            # --- [4] Cálculo do erro dos dados (E_D) ---
                            Err <- if(fwded[["costF"]] == "crossEntropy") 
                            {
                              -sum(ACTUAL * log(yhat) + (1 - ACTUAL) * log(1 - yhat))
                              
                            }else{ 0.5*sum((ACTUAL - yhat)^2)} # Para MSE
                            
                            
                            # --- [5] Cálculo de M(w) = β E_D(w₀) + α E_W(w₀)
                            # --- [5] Cálculo de M(w) = β E_D + α E_W (apenas W e b)
                            FinalErr <-   .ErrBayes(NET = fwded, NetErr = Err, sep = TRUE) 
                            Total    <-    FinalErr$edata + FinalErr$EP_all
                            Total
                               
                          }
  
  
  # --- Função de ∇M(w₀) ---
  gr1   <-   function(init.all)
                          {
                            # --- [1] Atualizar BLWNN com w₀
                            net      <-   .UpNetWts(wts = init.all, NET = NETWORK) 
                            
                            # --- [2] Propagação direta (feedforward)
                            network  <-   .ffwd(net)                        
                            
                            # --- [3] Derivadas das funções wavelets ψ'(a)
                            #___________________________________________________________________________________________________
                            # Limites para 'a' em sig_dev2:
                            # - Superior: -log(.Machine$double.eps) ≈ 36 (evita exp(a) = Inf)
                            # - Inferior: -log(1/.Machine$double.xmin - 1) ≈ -708 (evita exp(a) = 0)
                            if(network[["waveF"]] == "sig_dev2")   DactF  <- bquote({
                                                                                      a_tr <-  pmin(a, -log(.Machine$double.eps))        
                                                                                      a_tr <-  pmax(a_tr, -log(1/.Machine$double.xmin - 1)) 
                                                                                      sig  <- 1 / (1 + exp(-a_tr))                       
                                                                                      -6 * sig**4 + 12 * sig**3 - 7 * sig**2 + sig
                                                                                    })
                            
                            if(network[["waveF"]] == "mexicanhat") DactF  <- bquote({ a_tr <- pmin(pmax(a, -30), 30)
                                                                                      (a_tr**3 - 3*a_tr)*exp(-a_tr**2/2)})
                            #___________________________________________________________________________________________________
                            # --- [4] Obter informações relevantes da Propagação direta ---
                            u         <-  network[["wavePre"]]      # u = w1*x + b1
                            a         <-  network[["waveInput"]]      # a = (u.r) - t_tilde 
                            yhat      <-  network[["trainout"]]       # ŷ 
                            
                            
                            pa        <-  network[["waveOut"]]        # h = ψ(a) 
                            norm_data <-  network[["normData"]]       # entradas normalizadas
                            r         <-  exp(network[["weights"]][[4]])  # r = exp(log(r))
                            
                            if(any(!is.finite(r)) || any(r < 1e-8))
                              stop("Invalid dilation encountered")
                            
                            
                            # --- [5] Erro na saída (ŷ - y) ---
                            outError  <-  (yhat - ACTUAL)     
                            
                            
                            # --- [6] Derivadas da camada de saída (W2 e b2) ---
                            Gw2         <-  t(pa) %*% outError  # ∂C/∂W2 
                            Gb2         <-  colSums(outError)   # ∂C/∂b2 
                            
                            
                            # --- [7] Inicialização das listas para gradientes e erros de camada ---
                            # RCweights   -> (∂C/∂W1, ∂C/∂b1, ∂C/∂W2, ∂C/∂b2, ∂C/∂t_tilde, ∂C/∂log(r)) 
                            # layerErrors -> erros propagados
                            
                            RCweights          <-  list()
                            layerErrors        <-  list()
                            
                            
                            # --- [8] Armazenar δ₂ = (ŷ - y) como erro da camada 2 ---
                            layerErrors[[2]]   <- outError 
                            
                            
                            
                            # --- [9] Retropropagação para camada oculta (wavelons) ---
                            # erro da camada oculta: δ₁ = (δ₂ %*% t(W2)) * ψ'(a)
                            # eval(DactF) = ψ'(a) 
                            layerErrors[[1]]  <-  (layerErrors[[2]] %*% 
                                                     t(network[["weights"]][[2]][-nrow(network[["weights"]][[2]])])) * 
                                                     eval(DactF)
                            
                            
                            # ∂C/∂t_tilde = - δ₁
                            # r = exp(log(r)) = 1/s
                            RCweights[[3]]   <-  sweep(matrix(colSums(layerErrors[[1]]), ncol = 1), 
                                                       1, 
                                                       -1, 
                                                       FUN="*")
                            
                            # ∂C/∂log(r) = δ₁ * u * r
                            # r = exp(log(r)) = 1/s  
                            
                            RCweights[[4]]   <- matrix(colSums(sweep((layerErrors[[1]] * u),
                                                                     2, 
                                                                     r, 
                                                                     FUN="*")), ncol = 1) 
                          
                            # ∂C/∂W1 = t(x) %*% (δ₁ * r)
                            # r = exp(log(r))
                            RCweights[[1]]   <- t(norm_data) %*%  sweep(layerErrors[[1]], 
                                                                        2, 
                                                                        r, 
                                                                        FUN="*") 
                            
                            
                            # ∂C/∂b1 =  δ₁ * r
                            # r = exp(log(r))
                            Gb1              <-  colSums(sweep(layerErrors[[1]], 
                                                               2, 
                                                               r, 
                                                               FUN="*"))   
                               
                            
                            # --- [10] Agrupamento: ∂C/∂W1 e ∂C/∂b1 ---
                            RCweights[[1]] <-  rbind(RCweights[[1]], Gb1)  
                            
                            
                            # --- [11] Agrupamento: ∂C/∂W2 e ∂C/∂b2 --- 
                            RCweights[[2]] <-  rbind(Gw2, Gb2) 
                            
                          
                            
                            # --- [12] Cálculo de ∇M(w₀) = β * ∇E_D(w₀) + α * ∇E_W(w₀)
                            #                     ∂M(θ)/∂θ = β * ∂E_D/∂θ + α * ∂E_θ/∂θ
   
                            gdata_pure    <-  RCweights
                            
                            # β * ∇E_D(w₀)
                            # Se hp.Err > 0      --> β * ∇E_D(w₀)
                            # Se hp.Err = FALSE  --> ∇E_D(w₀)
                            if(is.numeric(network[["scale.error"]]) && network[["scale.error"]] > 0){
                                    gdata  <-  lapply(gdata_pure, "*", network[["scale.error"]]) 
                            } else {gdata  <-  gdata_pure}

                            
                            # --- Cálculo de α * ∇E_W(w₀) = ∂/∂W(α * w₀^2/2) = α * w₀  ---
                            if(length(network[["scale.weights"]][[1]]) == 1) 
                            { # α_W1 é um único valor 
                              alpha_w1b1 <- c(rep(network[["scale.weights"]][[1]], network$ncov), # α_w1 (1) (escalar)
                                              network[["scale.weights"]][[2]])                    # α_b1 (1)
                              
                            } else { # α_W1 específico p/ cda covariável
                              alpha_w1b1 <- c(network[["scale.weights"]][[1]],  # α_w1 (ncov) (vetor)
                                              network[["scale.weights"]][[2]])  # α_b1 (1)
                            }
                            # Validar a construção 
                            stopifnot(length(alpha_w1b1) == nrow(network[["weights"]][[1]]))
                          
                            
                            # α_w1 * ∇E_W1 = ∂/∂W1(α_w1 * w1^2/2) = α_w1 * w1
                            # α_b1 * ∇E_b1 = ∂/∂b1(α_b1 * b1^2/2) = α_b1 * b1
                            w1b1_pri  <-  sweep(network[["weights"]][[1]], 
                                                1, 
                                                alpha_w1b1, 
                                                FUN = "*")
                    
                            # Repetimos o mesmo hp.W2 para todos wavelons 
                            # α_w2 * ∇E_W2 = ∂/∂W2(α_w2 * w2^2/2) = α_w2 * w2
                            # α_b2 * ∇E_b2 = ∂/∂b2(α_b2 * b2^2/2) = α_b2 * b2
                            alpha_w2b2 <- c(rep(network[["scale.weights"]][[3]], network$hidden_size),# α_w2
                                            network[["scale.weights"]][[4]])                          # α_b2
                            
                            # Validar a construção 
                            stopifnot(length(alpha_w2b2) == nrow(network[["weights"]][[2]]))
                            
                            
                            w2b2_pri    <- sweep(network[["weights"]][[2]], 
                                                 1, 
                                                 alpha_w2b2, 
                                                 FUN = "*")

                            
                            
                            # Repetimos o mesmo hp.t.tilde para todos wavelons  
                            # α_t * ∇E_t = ∂/∂t(α_t * t^2/2) = α_t * t
                            t_tilde_pri  <-  network[["weights"]][[3]] * network[["scale.weights"]][[5]] # α_t_tilde
                            
                            if(identical(network$scale.weights[[5]], FALSE))
                              t_tilde_alpha  <- 0 * network$weights[[3]]
                            
                            
                            # Repetimos o mesmo valor hp.r para todos wavelons   
                            # α_log(r) * ∇E_log(r) = ∂/∂log(r)(α_log(r) * log(r)^2/2) = α_log(r) * log(r)
                            # Onde: E_log(r) = (1/2) * log(r)^2 (energia do prior)
                            log.r_pri  <-  network[["weights"]][[4]] * network[["scale.weights"]][[6]]   # α_log.r   
                            
                            # Gradientes a priori => α * ∇E_W(w) 
                            gprior  <-  list(w1b1_pri, 
                                             w2b2_pri,
                                             t_tilde_pri,
                                             log.r_pri)
                            
 
                            # Gradiente da função de custo regularizada  ∂M(θ)/∂θ 
                            # ∇M(w) = β * ∇E_D(w) + α * ∇E_W(w)
                            GradTotal  <-  mapply("+", gdata, gprior)
                        

                            # --- [13] Reorganização dos gradientes em vetor único para otimização ---
                            grad_w1b1    <-  c(t(GradTotal[[1]])) # (β*∂C/∂W1 + α_w1 * w1, β*∂C/∂b1 + α_b1 * b1) 
                            grad_w2b2    <-  c(t(GradTotal[[2]])) # (β*∂C/∂W2 + α_w2 * w2, β*∂C/∂b2 + α_b2 * b2) 
                            grad_t_tilde <-  c(t(GradTotal[[3]])) # (β*∂C/∂t_tilde + α_t_tilde * t_tilde) 
                            grad_log.r   <-  c(t(GradTotal[[4]])) # (β*∂C/∂log.r + α_log.r * log.r) 
                            
                            # --- [14] Retornar vetor gradiente para o otimizador ---
                            VecDeriv  <-  c(grad_w1b1, grad_w2b2, grad_t_tilde, grad_log.r)
                            return(VecDeriv)
                            
                          }

  
  # --- Otimização dos parâmetros via método BFGS ---
  output  <-  optim(init.all,                     # Parâmetros iniciais (w_0)
                    fn1,                          # Custo M(w) a ser minimizado
                    gr1,                          # Gradiente ∇M(w)
                    method = "BFGS",              # Algoritmo de otimização
                    hessian = TRUE,               # Matriz Hessiana aproximação de A = H + αI = β*∇∇E_D(w) + αI
                    control = list(maxit = iter)) # Número máx. de iterações
  
  return(output)
}

