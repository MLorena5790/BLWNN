# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .evidence_all
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
# Purpose: Evidence-based estimation of hyperparameters
#          associated with all BLWNN parameter groups.
#
# Contribution: Adapted and substantially extended from
#               the original BLNN implementation to
#               incorporate evidence-based estimation of
#               prior precision hyperparameters associated
#               with reparameterized wavelet translation
#               and dilation parameters.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.evidence <- function(NET,         # objeto BLWNN
                      actual,      # saidas / respostas reais/ verdadeiras 
                      itter = 5,   # n. de iterações de evidência
                      seed = NULL)
  
{
  
  # --- GARANTIR REPRODUTIBILIDADE DA OTIMIZAÇÃO BFGS ---
  if(!is.null(seed)) set.seed(seed)
  
  #------------------------------------------------------
  # Evidência calculada apenas para (W1, b1, W2, b2, t_tilde, log(r)).
  #------------------------------------------------------
   
  net            <-  NET
  act            <-  actual
  norm_data      <-  net[["normData"]]         # entradas normalizadas
 
  
  # --- [1] Estimação do MAP e obtenção da Hessiana total via BFGS (não-bayesiano) ---
  opt_result     <-  .BLWNN_BFGS_all(NET = net, actual = act) # Minimiza M(w) 
  all_MAP        <-  opt_result$par                           # Estimativa w_MAP = (W1, W2, b1, b2, t_tilde, log(r))
  hess_total     <-  opt_result$hessian                       # Hessiana total A ≈ ∇²M(w_MAP) = β*∇∇E_D(w_MAP) + αI
  
  
  
  # --- [2] Calculando a hessiana H = ∇∇E_D(w_MAP) ---
      # --- [2.1] Construção do vetor α (hiperparâmetros de regularização) ---
      if(length(net$scale.weights[[1]]) == 1)
      {       alpha_w1  <-  rep(net[["scale.weights"]][[1]], net$ncov * net$hidden_size) # α_W1 é um único valor 
      } else{ alpha_w1  <-  rep(net[["scale.weights"]][[1]], each = net$hidden_size)}    # α_W1 específico p/ cd covariável
      
      alpha_vec  <- c(alpha_w1,                                      # α_W1
                      rep(net$scale.weights[[2]], net$hidden_size),  # α_b1
                      rep(net$scale.weights[[3]], net$hidden_size),  # α_w2
                      rep(net$scale.weights[[4]], net$nout),         # α_b2
                      rep(net$scale.weights[[5]], net$hidden_size),  # α_t_tilde
                      rep(net$scale.weights[[6]], net$hidden_size))  # α_log(r)
      
      # --- [2.2] Hessiana dos dados: H = ∇∇E_D(w_MAP)  ---
      hess_data      <-  hess_total - diag(alpha_vec)     #     H_β = β ∇∇E_D(w_MAP)
                                                          # autovalores ( λ₁ = βμ₁, λ₂ =  βμ₂, ..., λₘ = βμₘ)
      hess_data_pure <-  hess_data / net[["scale.error"]] # H_β / β = ∇∇E_D(w_MAP)
                                                          # autovalores (μ₁,μ₂, ...,μₘ)   
      
      
  # --- [3] Cálculo de E_D(w_MAP) e E_W(w_MAP) ---
      # --- [3.1] Atualizar BLWNN_Build com w_MAP = (W1, W2, b1, b2, t_tilde, log(r))
      #net_w_MAP    <-  .UpNetWbOnly(wts = Wb_MAP, NET = net) 
      net_w_MAP    <-  .UpNetWts(wts = all_MAP, NET = net) 
      
      # --- [3.2] Erros puros: E_D(w_MAP) e E_W(w_MAP) por grupo
      #allerr       <-  .NetErr_Wb(Actual = act, NET = net_w_MAP, Evidence = TRUE)
      allerr       <-  .NetErr(Actual = act, NET = net_w_MAP, sep = TRUE)
      
      edata_pure   <-  allerr$Data      # E_D(w_MAP) 
      eprior_pure  <-  allerr$EVEP_all  # E_W(w_MAP)  = 1/2·w²_MAP separada por grupo W1, b1, W2, b2, t_tilde, log(r)
                                        # eprior_pure = c( ∑_j (1/2)·w1_j², 
                                        #                  ∑ (1/2)·b1², 
                                        #                  ∑_j (1/2)·w2_j², 
                                        #                  ∑ (1/2)·b2² )
                                        #                  ∑ (1/2)·t_tilde²
                                        #                  ∑ (1/2)·log(r)²

      
  # --- [4] Análise Espectral da Hessiana H_pure = ∇∇E_D(w_MAP) ---
  eigDH     <-  eigen(hess_data_pure, symmetric = TRUE)
  evec      <-  eigDH$vectors # autovetores 
  evl       <-  eigDH$values  # Autovalores (μ₁,μ₂, ...,μₘ)
  nparam    <-  length(evl)   # M = Total de parâmetros w_MAP = (W1, b1, W2, b2, t_tilde, log(r))
   
  
  # --- [5] Correção de μ₁,μ₂, ...,μₘ para aproximação de Laplace (positivos e ≠ 0) ---
  min_evl        <- sqrt(.Machine$double.eps)                    # Valor mínimo seguro (√ε ≈ 1.5e-8)
  evl_correct    <- pmax(evl, 1e-8)                              # Garante μ ≈ 0 sejam μ ≥ √ε

  
  # --- [6] Preparação para cálculo de γ --
  # Número total de hiperparâmetros α distintos 
  # ngroup = #(α_w1) + #(α_b1) + #(α_W2) + #(α_b2) + #(α_t_tilde) + #(α_log(r))
  ngroup    <-  length(net_w_MAP$scale.weights[[1]])+ #length(α_W1)
                length(net_w_MAP$scale.weights[[2]])+ #length(α_b1)
                length(net_w_MAP$scale.weights[[3]])+ #length(α_W2)
                length(net_w_MAP$scale.weights[[4]])+ #length(α_b2)
                length(net_w_MAP$scale.weights[[5]])+ #length(α_t_tilde)
                length(net_w_MAP$scale.weights[[6]])  #length(α_log(r))
    
  
 
  # --- [7] Inicialize o contador  --- 
  counter       <-  0

  while(counter < itter)
  {
    
    # --- [7.1] Número de parâmetros (M_i) por grupo: W1, b1, W2, b2, t_tilde, log(r) ---
    if(length(net_w_MAP$scale.weights[[1]]) == 1)
    {       n_in_groups_W1 <- net_w_MAP$ncov * net_w_MAP$hidden_size       # α_W1 é um único valor 
    } else{ n_in_groups_W1 <- rep(net_w_MAP$hidden_size, net_w_MAP$ncov) } # α_W1 específico p/ cd covariável

    stopifnot(length(n_in_groups_W1) == length(net_w_MAP$scale.weights[[1]]))
    
    # ngroup = c(#(w1), #(b1), #(W2), #(b2), #(t_tilde), #(log(r)))  
    n_in_groups  <-  c(n_in_groups_W1,                       # length(W1) 
                       net_w_MAP$hidden_size,                # length(b1) 
                       net_w_MAP$hidden_size*net_w_MAP$nout, # length(W2) 
                       net_w_MAP$nout,                       # length(b2) 
                       net_w_MAP$hidden_size,                # length(t_tilde) 
                       net_w_MAP$hidden_size)                # length(log(r)) 
    
    
    
    # --- [7.2] Hiperparâmetros atuais por grupo: W1, b1, W2, b2, t_tilde, log(r) ---
    # alpha_group = c(α_W1, α_b1, α_W2, α_b2, α_t_tilde, α_log(r))
    # OBS.: alpha_group é atualizado a cada iteração em net_w_MAP
    alpha_group  <-  c(net_w_MAP$scale.weights[[1]], # α_W1 (escalar ou vetor)
                       net_w_MAP$scale.weights[[2]], # α_b1
                       net_w_MAP$scale.weights[[3]], # α_W2
                       net_w_MAP$scale.weights[[4]], # α_b2
                       net_w_MAP$scale.weights[[5]], # α_t_tilde
                       net_w_MAP$scale.weights[[6]]) # α_log(r)


    # --- [7.3] Validação (ngroup e alpha_group)
    if(length(n_in_groups) != length(alpha_group)) 
      stop("length of first layer hyperparameters does not match 
                       number of covariates or is not length 1")
    

    # --- [7.4] Calcular trace(A⁻¹)_i = ∑ diag(A⁻¹)_i por grupo
    # Soma da diagonal principal A⁻¹ = (β*∇²E_D + αI)⁻¹ por grupo
    # Ou Nabney(2002, pag. 344) trace(A⁻¹)_i = ∑_i 1 / ( βμ_i + α_i) por grupo
    # OBS.: net_w_MAP[["scale.error"]] é atualizado a cada iteração

    group_ids         <- rep(seq_along(n_in_groups), times = n_in_groups)                     # particionar ∑ por grupo
    alpha_equiv       <- rep(alpha_group, times = n_in_groups)                                # α_group_i, um α p/ cda autovalor μ_i
    HessDiagGroupSum  <- tapply(1 / (net_w_MAP[["scale.error"]] * evl_correct + alpha_equiv), # trace(A⁻¹)_i = ∑_i 1 / ( βμ_i + α_i)
                                group_ids, 
                                sum)                                                         
    
    # --- [7.5] Gamma (complexidade) por grupo => γ_i = M_i - α_i * trace(A⁻¹)_i
    group_gamma                  <- n_in_groups - alpha_group * HessDiagGroupSum
    group_gamma[group_gamma < 0] <- 0
    
    # --- [7.6] Atualização de α por grupo => α_new_i = γ_i / (2 * E_W_i) por grupo
    alpha_group_new                           <-  Re(group_gamma / (2*eprior_pure))  
    alpha_group_new[alpha_group_new <= 1e-10] <-  1e-10
    
    
    # --- [7.7] Atualização de β => β_new = (N_eff - γ_total) / (2 * E_D(w_MAP))
    gamma       <-   sum(group_gamma)                   # γ_total = ∑ γ_i
    N_eff       <-   net_w_MAP$nout * nrow(norm_data)   # n. saídas que contribuem E_D
    beta_new    <-   (N_eff  - gamma)/(2*edata_pure)    # β = (N_eff-γ)/(2E_D)
    beta_new    <-   max(1e-10, beta_new)
    
    
    # --- [7.8] 1º parte da log-evidência log p(D|α,β) =>  (1/2)∑ M_i·log(α_new_i) + (1/2)·N·log(β_new) - (1/2)·N·log(2π)
    logas       <-   .5 * n_in_groups * log(alpha_group_new)   #   (1/2).M_i·log(α_new_i)
    logev       <-   sum(logas)                                # ∑ (1/2).M_i·log(α_new_i)
    logev       <-   logev + 
                     .5 * N_eff * log(beta_new) - # + (1/2)·N·log(β_new)
                     .5 * N_eff * log(2*pi)       # - (1/2)·N·log(2π)
    

    # --- [7.9] Atualização de α_i_new e β_new na rede c/ w_MAP = (W1, W2, b1, b2, t_tilde, log(r))
    net_w_MAP$scale.error        <-  beta_new  # β_new
    
    net_w_MAP$scale.weights[[1]] <- alpha_group_new[1:length(net_w_MAP$scale.weights[[1]])] # α_W1_new
    net_w_MAP$scale.weights[[2]] <- alpha_group_new[ngroup - 4]                             # α_b1_new
    net_w_MAP$scale.weights[[3]] <- alpha_group_new[ngroup - 3]                             # α_W2_new
    net_w_MAP$scale.weights[[4]] <- alpha_group_new[ngroup - 2]                             # α_b2_new
    net_w_MAP$scale.weights[[5]] <- alpha_group_new[ngroup - 1]                             # α_t_tilde_new
    net_w_MAP$scale.weights[[6]] <- alpha_group_new[ngroup]                                 # α_log(r)_new
                                  
    
    # --- [7.10] Atualização da contagem 
    counter     <-   counter + 1
  }
  
  
  
  # --- [8] 2º parte da log-evidência log p(D|α,β) => - α_new*E_W(w_MAP) - β_new*E_D(w_MAP) - 0.5*log(det(A))
  
      # --- [8.1] Cálculo de M(w_MAP) = β_new*E_D(w_MAP) + α_new*E_W(w_MAP)   
      error   <-   .NetErr(Actual = act, NET = net_w_MAP, sep = FALSE)
  
      # --- [8.2] Cálculo de log(det(A)) = ∑_i log(β_new * μ_i + α_new) 
      alpha_equiv     <-  rep(alpha_group_new, times = n_in_groups)         # α_new para cada grupo
      log_detA        <-  sum(log(beta_new * evl_correct + alpha_equiv))    # log(det(A)) = ∑_i log(β_new * μ_i + α_new) 
      log_detA        <-  max(log_detA, .Machine$double.eps)
      

      # --- [8.3] Cálculo completo de log p(D|α,β) 
      # log p(D|α,β) = (1/2)∑ M_i·log(α_new_i) + (1/2)·N·log(β_new) - (1/2)·N·log(2π) - 
      #                 β_new*E_D(w_MAP) - α_new*E_W(w_MAP) - 0.5*log(det(A)) 
      logev_complete   <- logev - error - 0.5 * log_detA
  
  return(list(alpha_vals   = net_w_MAP$scale.weights,# (α_new_W1, α_new_b1, α_new_W2, α_new_b2)
              beta_val     = beta_new,               # β_new
              log_evidence = logev_complete))        # log p(D|α_new,β_new)

}
