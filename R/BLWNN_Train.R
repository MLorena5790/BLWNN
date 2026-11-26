# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .BLWNN_Train
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
# Purpose: Bayesian wavelet neural network training
#          through posterior sampling using HMC and NUTS.
#
# Contribution: Adapted and substantially extended from
#               the original BLNN implementation to
#               accommodate reparameterized wavelet
#               parameters, parameter-specific blocks,
#               and optional evidence-based hyperparameter
#               estimation within the BLWNN framework.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.BLWNN_Train <-  function( 
                          NET,                             # objeto .BLWNN_Build
                          y,                               # respostas verdadeiras y
                                                           #______________________________________________________
                                                           # ----- Parâmetros HMC -----
                          iter        = 2000,              # (HMC) -> n. de amostras extraídas da dist. a posteriori
                                                           #          n. de iterações do algoritmo HMC
                          init        = NULL,              # (HMC) -> lista de vetores com valores iniciais dos parâmetros 
                                                           #          (w1, b1, w2, b2, t_tilde, log(s)) ou uma função 
                          chains      = 2,                 # (HMC) -> n. de cadeias de Markov a serem executadas 
                          seeds       = NULL,              # (HMC) -> vetor de semente --> 01 para cada cadeia 
                          warmup      = NULL,              # (HMC) -> n. de iterações/amostras de aquecimento 
                                                           #          geralmente, metade do n. de iterações  
                                                           #          DA -> ameniza o problema de ajustar eps  
                          thin        = 1,                 # (HMC) -> taxa de diluição a aplicar às amostras 
                          parallel    = FALSE,             # valor booleano para verificar se deve usar núcleos 
                                                           # paralelos ou não. 
                          cores       = NULL,              # n. de núcleos a serem usados se parallel = TRUE
                          algorithm   = c("NUTS", "HMC"),  # Algoritmos -> NUTS para No-U-Turn. 
                                                           #               HMC para hamiltoniano 
                                                           #______________________________________________________
                                                           # ----- Evidência para α (pesos e vieses) e β -----
                          evidence      = FALSE,           # Procedimento de evidência para reestimar α e β. 
                          evidence_y    = NULL,            # Vetor/matriz de respostas verdadeiras y usados na evidência.
                          evidence_iter = 5,               # N. de iterações no procedimento de evidência. 
                                                           #______________________________________________________
                                                           #
                          control     = NULL,              # lista com argumentos de controle necessários para 
                                                           # ajustar NUTS e HMC 
                                                           #    adapt_delta   -> taxa de aceitação do alvo. HMC -> 0.65
                                                           #    momentum.mass -> vetor de variância do momento
                                                           #    stepsize      -> tamanho do passo eps
                                                           #                     NULL --> será adaptado durante aquecimento 
                                                           #                     UseDA = T, stepsize é o passo inicial usado 
                                                           #                                em Pick Epsilon. 
                                                           #    useDA         -> se a média dupla para adaptar o 
                                                           #                     tamanho do passo é usada ou não. 
                                                           #    gamma         -> um dos argumentos DA. 
                                                           #    t0            -> um dos argumentos DA
                                                           #    kappa         -> um dos argumentos DA 
                                                           #    metric        -> a métrica de massa a ser usada 
                                                           #    adapt_mass    -> se a métrica da matriz de massa é ativada. 
                                                           #    w1            -> inteiro positivo
                                                           #    w2            -> inteiro positivo 
                                                           #    w3            -> inteiro positivo 
                                                           # Argumentos usados apenas para NUTS:
                                                           #    max_treedepth -> intiro postivo 
                                                           # Argumentos usados apenas para HMC
                                                           #    lambda        -> comprimento de simulação de uma trajetória. 
                                                           # param_blocks     -> índices dos blocos de parâmetros (opicional)
                          
                          display     = 0,                 # ajuda a rastrear o algoritmo do amostrador exibindo 
                                                           # vários resultados: 
                                                           # 0 --> não exibe nada
                                                           # 1 --> exibe o erro da rede após cada iteração 
                                                           # 2 --> (HMC) stepsize e n. de passos durante e após aquecimento 
                                                           # 3 --> (HMC) tbm exibe energia antiga e nova para cada iteração
                          path        = getwd(),                    
                          ...) 
{
  
  # --- [1] Verificar a existência do vetor y ---
  if (is.null(y)) 
    stop("Make sure network's input and target are assigned")
  
  # --- [2] Verificar se NET é .BLWNN_Build (carrega informações primordiais) ---
  if (class(NET) != "BLWNN")
    stop("Make sure your Network is a BLNN object, use BLNN_Build to create the BLWNN object.")
  
  # --- [3] Inicializar variáveis principais ---
  Actual      <-  y
  inputs      <-  NET[["normData"]] # entradas normalizadas
  
  
      # --- [3.1] Definição automática do warmup (n. de amostras de aquecimento ) ---
      # Durante o warmup, a cadeia não amostra a posteriori, mas calibra quantidades importantes:
      #   - stepsize 
      #   - matriz de massa 
      #   - estatísticas internas do sampler
      if(is.null(warmup)) 
        warmup    <- floor(iter / 2)    
      
  
  #____________________________________________________________________________________________________
  # --- [4] Reestimar α e β pelo procedimento de evidência antes da amostragem HMC/NUTS ---
  # Este bloco executa o procedimento de maximização da evidência com aproximação Laplaciana
  # aplicada apenas a (W1, b1, W2, b2), mantendo parâmetros wavelets fixos.
  if(evidence) 
  {
    # [4.1] Definição da resposta: usa as respostas de treinamento se evidence_y = NULL
    if(is.null(evidence_y))    evidence_y = Actual
    
    # [4.2] Reestimação de α (restrito W1, b1, W2, b2) e β 
    evidence_out      <- .evidence(NET    = NET, 
                                   actual = evidence_y, 
                                   itter  = evidence_iter) 

    # [4.3] Atualização de β na rede
    NET$scale.error   <- evidence_out$beta_val         
    
    # [4.4] Atualização de α = (α_W1, α_B1, α_W2, α_B2) 
    NET$scale.weights <- evidence_out$alpha_vals 
  }
  
  # Com α e β calibrados, o HMC amostra apenas os parâmetros. α e β são fixos daqui em diante.
  #____________________________________________________________________________________________________
  
  
  # --- [5] Definição de variáveis ---
  n.params      <- length(.VecNetWts(NET))       # dimensão do espaço paramétrico 
                                                 # #(W1) + #(b1) + #(W2) + #(b2) + #(t_tilde) + #(log(s))
  par.names     <- as.character(c(1:n.params))   # gerar rótulos artificiais p/ cda parâmetro 

  # --- [5.1] Definição dos blocos de parâmetros ---
  bkpt1      <-  (NET[["ncov"]] + 1) * NET[["hidden_size"]] # n. pesos de entrada + b1  =  (ncov + 1) * n_wavelons
  bkpt2      <-  (NET[["hidden_size"]] + 1) * NET[["nout"]] # n. pesos de saída + b2 = (n_wavelons + 1) * n_outputs
  bkpt3      <-  NET[["hidden_size"]]                       # n. translações (1 por wavelon)
  bkpt4      <-  NET[["hidden_size"]]                       # n. log(s) (1 por wavelon)
  
  stopifnot(n.params == bkpt1 + bkpt2 + bkpt3 + bkpt4)
  
  idx_blocks           <- list( weights     = c(1:(bkpt1+bkpt2)),                           # idx_(W1, b1, W2, b2)
                                  trans     = ((bkpt1+bkpt2) + 1):((bkpt1+bkpt2) + bkpt3),# idx_t_tilde
                                  log_dilat = ((bkpt1+bkpt2+bkpt3) + 1): n.params)        # idx_og(s)
  
  control$param_blocks <- idx_blocks
  

  # --- [6] Função auxiliar para cálculo da energia potencial 
  #         M(θ) = U(θ) = - log P(θ|y) = −log P(y|θ) − log P(θ)  ---
  U      <- function(NewWts) 
  {
    # [6.1] Atualizar os parâmetros θ = (W1, b1, W2, b2, t_tilde, log(s)) 
    UpNet     <-  .UpNetWts(wts = NewWts, # vetor de parâmetros unificado (.VecNetWts)
                            NET = NET)   
    # [6.2] Propagação direta (feedforward)
    UpNetFwd  <-  .ffwd(UpNet)              
    # [6.3] Calcular M(θ) = β.E_D(θ) + α.E_W(θ)
    UpErr     <-  .NetErr(Actual = Actual, 
                          NET    = UpNetFwd, 
                          sep    = FALSE) 
    UpErr
  }

  
  
  # --- [7] Função auxiliar para cálculo do gradiente energia potencial 
  #         ∇M(θ) = ∇U(θ) = - ∇log P(θ|y) ---
  grad.U <- function(NewWts) 
  {
    # [7.1] Atualizar os parâmetros θ = (W1, b1, W2, b2, t_tilde, log(s)) 
    UpNet      <-  .UpNetWts(wts = NewWts, 
                             NET = NET)     
    # [7.2] Cálculo o gradiente ∇M(θ) = β * ∇E_D(θ) + α * ∇E_W(θ)
    UpDeriv    <-  .derivs(Actual = Actual, 
                           NET    = UpNet)    
    # [7.3] Reorganizar os gradientes no espaço vetorial (onde HMC opera)
    VecDeriv   <-  unlist(UpDeriv)
    VecDeriv
  }
  
  
  
  
  # --- [8] Verificação das sementes p/ inicialização de cda cadeia ---
  #         Controle da inicialização do processo estocástico 
  # [8.1] seeds == NULL, gerar uma semente aleatória distinta p/ cda cadeia. 
  if(is.null(seeds)) {
    seeds <- as.integer(runif(chains, 1, 100000))
    
  # [8.2]  Verificar se o n. de sementes = n. de cadeias. 
  } else if(length(seeds) != chains) {
    stop("O comprimento das sementes não é igual ao número de cadeias.")
  
  # [8.3] Garantir sementes diferentes p/ evitar inicializações repetidas.  
  } else if(anyDuplicated(seeds)) {
    stop("As sementes fornecidas contêm valores repetidos.")
  }
  
  


  # --- [9] Inicialização de θ = (W1, b1, W2, b2, t_tilde, log(s)) p/ as cadeias de amostragem ---
  #         Controle do início das cadeias no espaço paramétrico 
  if(is.null(init)) {
  # [9.1] init == NULL, usar msm valores iniciais de .BLWNN_Build p/ tds as cadeias 
  if(chains > 1)
    warning('Usando os mesmos valores iniciais para cada cadeia  -- 
                     é fortemente recomendado usar inicializações dispersas.')
  init <- lapply(1:chains, function(i) as.numeric(.VecNetWts(NET)))  

  # [9.2] init == função, executar a função p/ gerar inicializações.   
  #       init_function => Gera objetos .BLWNN_Build c/ inicializações mediante sementes diferentes, 
  #                        trajetórias diferentes. 
  } else if(is.function(init)) { init <- lapply(1:chains, function(i) init(NET, seeds[i]))
    
  # [9.3] Verificar se o nº de inicializações == nº de cadeias. 
  } else if(length(init) != chains) {
    stop("O comprimento do init não é igual ao número de cadeias.")
    
  # [9.4] Verificar se o nº de parâmetros inicializados = nº de parâmetros da rede
  } else if(any(unlist(lapply(init, function(x) length(unlist(x)) != n.params)))) { 
    stop("O vetor de parâmetro inicial tem comprimento incorreto")}

  

  # --- [10] Validar o argumento 'algorithm' ---
  algorithm    <-  match.arg(algorithm, choices = c("NUTS", "HMC"))
  
  
  
  # --- [11] Verificar se 'iter' é numérico e maior que 10 ---
  # 'iter' = nº de amostras extraídas da posteriori durante a amostragem bayesiana
  if(iter < 10 | !is.numeric(iter))
    stop("iter deve ser > 10") 
  
  ######################
  ######################
  
  
  
  mcmc.out <- list()
  
  #Verifique se há paralelo 
  if(!parallel) {
    if(algorithm == "HMC") {
      mcmc.out  <-  lapply(1:chains, function(i) #substitua isso pela nossa função .sample_NN_hmc
        .sample_NN_hmc( iter    = iter,
                        fn      = U,
                        gr      = grad.U,
                        init    = init[[i]],
                        warmup  = warmup,
                        chain   = i,
                        thin    = thin,
                        seed    = seeds[i],
                        control = control, 
                        display)
      )
    } else if (algorithm == "NUTS") {
      mcmc.out  <-  lapply(1:chains, function(i) #substitua isso pela nossa função .sample_NN_nuts
        .sample_NN_nuts(  iter    = iter,
                          fn      = U,
                          gr      = grad.U,
                          init    = init[[i]],
                          warmup  = warmup,
                          chain   = i,
                          thin    = thin,
                          seed    = seeds[i],
                          control = control, display)
      )
      
      
    }
    
    
    #Execução paralela 
  } else {
    if (!requireNamespace("future.apply", quietly = TRUE))
      stop("future.apply package not found")
    
    
    if (file.exists('mcmc_progress.txt'))
      trash               <-   file.remove('mcmc_progress.txt')
    tempDir             <-   tempfile()
    dir.create(tempDir)
    htmlFile            <-   file.path(tempDir, 
                                       "mcmc_progress.txt")
    viewer              <-   getOption("viewer")
    #viewer(htmlFile)
    
    future::plan(future::multiprocess, workers = cores)
    message("Starting parallel chains... ")
    ##mcmc.out <- lapply(1:chains, function(i)
    
    
    if(algorithm == "HMC")
    {
      mcmc.out <- future.apply::future_lapply(1:chains, function(i) #substitua isso pela nossa função 
        #.sample_NN_hmc
        .sample_NN_hmc( iter    = iter,
                        fn      = U,
                        gr      = grad.U,
                        init    = init[[i]],
                        warmup  = warmup,
                        chain   = i,
                        thin    = thin,
                        seed    = seeds[i],
                        control = control, 
                        display, 
                        parallel, 
                        file = htmlFile,
        )
      )
    } else if (algorithm == "NUTS") {
      mcmc.out <- future.apply::future_lapply(1:chains, function(i) #substitua isso pela nossa função 
        # .sample_NN_nuts
        .sample_NN_nuts(  iter    = iter,
                          fn      = U,
                          gr      = grad.U,
                          init    = init[[i]],
                          warmup  = warmup,
                          chain   = i,
                          thin    = thin,
                          seed    = seeds[i],
                          control = control, 
                          display, 
                          parallel, 
                          file = htmlFile,
        )
      )
    }
    
    message("... Finished parallel chains")
    
  }
  
  
  # Aquecimento
  # mcmc.out[[1]]$warmup --> n. de iterações de aquecimento/ taxa de diluição --> warmup / thin
  warmup    <-  mcmc.out[[1]]$warmup
  
  
  # Limpar saída retornada 
  iters     <-    unlist(lapply(mcmc.out, 
                                function(x) dim(x$par)[1])) 
  # dim(x$samples)[1] --> n. de linhas --> 
  # n. de amostras da posteriori --> n. iterações HMC
  
  
  if(any(iters != iter/thin))
  {
    N          <-    min(iters)
    warning(paste("Variable chain lengths, truncating to minimum =", N))
    
  } else {
    N           <-    iter/thin
  }
  
  
  # Criando uma lista de "1+length(par.names)" arrays vazios --> "N" linhas x "chains" colunas 
  # array 3 dimensões -> iterações x cadeias x parâmetros 
  samples     <-    array(NA, 
                          dim      = c(N, chains, 1 + length(par.names)),
                          dimnames = list(NULL, NULL, c(par.names,'Er__')))
  
  
  # mcmc.out[[i]]$par --> theta.out --> current.q --> variáveis amostrada pelo HMC
  for(i in 1:chains)
    samples[,i,] <-    mcmc.out[[i]]$par[1:N, ]
  
  
  
  # Antes de transformar, obtenha a covariância estimada para ser usada 
  # como métrica posteriormente
  covar.est    <-    cov(do.call(rbind, 
                                 lapply(1:chains, 
                                        function(i)
                                          mcmc.out[[i]]$par[-(1:warmup), 1:n.params])))
  
  dimnames(covar.est) <- NULL
  
  message("... Calculando ESS e Rhat")
  
  # rstan::monitor --> usa a matriz de simulações 3d (samples) para calcular medidas  
  # resumo, como quantis. Por default --> warmup --> metade das iterações são consideradas 
  # aquecimento e são excluídas ao calcular os resumos. 
  temp                <-  (rstan::monitor(  samples,
                                            warmup = warmup,
                                            probs  = .5,
                                            print  = FALSE))
                                          
  Rhat                <-  temp[, 6]
  ess                 <-  temp[, 5]
  
  if(algorithm == "NUTS")
  {
    sampler_params  <-  lapply(mcmc.out, function(x) x$sampler_params[1:N,])
    
  } else {
    sampler_params  <-  lapply(mcmc.out, function(x) x$sampler_params[1:N,])
  }
  
  
  time.warmup      <-  unlist(lapply(mcmc.out, function(x) as.numeric(x$time.warmup)))
  time.total       <-  unlist(lapply(mcmc.out, function(x) as.numeric(x$time.total)))
  #cmd              <-  unlist(lapply(mcmc.out, function(x) x$cmd))
  
  
  if(N < warmup) warning("Duração muito curta para terminar o período de aquecimento")
  
  result <- list( samples        = samples,
                  sampler_params = sampler_params,
                  time.warmup    = time.warmup,
                  time.total     = time.total,
                  algorithm      = algorithm,
                  warmup         = warmup,
                  covar.est      = covar.est,
                  Rhat           = Rhat,
                  ess            = ess,
                  hp.list        = list(hp.W  = NET$scale.weights, 
                                        hp.E  = NET$scale.error))
  
  
  if(algorithm == "NUTS")
  {
    result$max_treedepth <- mcmc.out[[1]]$max_treedepth
    return(invisible(result))
  }
  
}
