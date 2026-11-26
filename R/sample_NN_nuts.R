# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .sample_NN_nuts
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
# Purpose: No-U-Turn Sampler (NUTS) posterior sampling
#          for Bayesian wavelet neural networks.
#
# Contribution: Adapted and extended from the original
#               BLNN implementation to support posterior
#               sampling within the BLWNN framework,
#               including step-size and metric adaptation,
#               numerical safeguards, parameter-space
#               transformations, and sampler diagnostics.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================

.sample_NN_nuts <-  function(iter,                            # n. de amostras extraídas da dist. posteriori 
                             fn,                              # Função auxiliar: calcula o custo regularizado 
                                                              # U --> Energia potencial do hamiltoniano
                             gr,                              # Função auxiliar: calcula o gradiente da rede 
                                                              # U' --> derivada da energia potencial 
                             init,                            # lista de vetores contendo os valores iniciais 
                                                              # dos parâmetros ou uma função -> i in 1:chains
                             warmup    = floor((iter) / 2),   # n. de iterações/amostras de aquecimento
                                                              # metade do n. de iterações (iter)
                             chain     = 1,                   # i corresponde a cadeia que será executada 
                             thin      = 1,                   # taxa de diluição a aplicar às amostras / diminuir correlação entre amostras
                             seed      = NULL,                # vetor de semente/ 01 para cada cadeia 
                             control   = NULL,                # lista com argumentos de controle necessários 
                                                              # para ajustar NUTS e HMC 
                                                             
                                                              #    adapt_delta   -> taxa de aceitação do alvo. HMC -> 0.65
                                                              #    momentum.mass -> vetor de variância do momento na dinâmica Hamiltoniana
                                                              #    stepsize      -> tamanho do passo a ser usado 
                                                              #    useDA         -> se a média dupla para adaptar o tamanho do passo 
                                                              #                     é usada ou não. 
                                                              #        gamma         -> um dos argumentos DA. 
                                                              #        t0            -> um dos argumentos DA. parâmetro livre que 
                                                              #                         estabiliza as iterações iniciais 
                                                              #        kappa         -> um dos argumentos DA 
                                                              #    metric        -> a métrica de massa a ser usada 
                                                              #    adapt_mass    -> se a métrica da matriz de massa é ativada. 
                                                              # Janelas para adaptação da matriz de massa (warmup dividido em janelas).
                                                              #    w1            -> inteiro positivo
                                                              #    w2            -> inteiro positivo 
                                                              #    w3            -> inteiro positivo 
                             display   = 0,                   # ajuda a rastrear o algoritmo do amostrador exibindo vários resultados
                             path      = getwd(),             # caminho para salvar resultados, se necessário.
                             parallel  = FALSE,               # valor booleano para verificar se deve executar as cadeias em núcleos paralelos ou não.
                             f1        = NULL,
                             ...) 
{
  
    # --- [2] Define parâmetros de controle para HMC/NUTS ---
    control <- .update_control(control)
    
    
    # --- [3] Define semente para reprodutibilidade HMC/NUTS ---
    if (!is.null(seed)) set.seed(seed)
    
    
    
    ##########################################################################################
    # Parâmetros que regem o comportamento do HMC/NUTS (baseado em Gelman e Hoffman)
    # Estratégia:
    # 1. Iniciamos com valores iniciais.
    # 2. Executamos o HMC com aquecimento (warmup).
    # 3. Durante o warmup, ajustamos parâmetros adaptativos.
    # 4. Após o warmup, fixamos os parâmetros e descartamos essas amostras.
    ##########################################################################################
    
    
    adapt_delta    <- control$adapt_delta     
    M              <- control$momentum.mass  
    useDA          <- control$useDA 
    
    gamma          <- control$gamma           
    t0             <- control$t0              
    kappa          <- control$kappa  
    
    init.eps       <- control$stepsize  
    init           <- as.vector(unlist(init)) # vetoriza valores iniciais dos parâmetros da rede
    npar           <- length(init)            # N. total de parâmetros (w1, b1, w2, b2, t, log(s))
    
    
    # --- [4]  Ajustes internos antes da simulação ---
    M_inv          <- 1 / M                   # Inverso da matriz de massa M -> uma matriz diagonal 
                                              # - Usado na energia cinética do HMC
                                              # - Se M é escalar → massa igual para todos os parâmetros
                                              # - Se M é vetor → pré-condicionamento por dimensão
    max_td         <- control$max_treedepth   # Profundidade máxima da árvore NUTS
                                              # - Define o limite de expansões do leapfrog
                                              # - Evita trajetórias muito longas e custo excessivo
    
    # [4.1] Preparação para adaptação da matriz de massa 
    M1                  <- control$metric     # métrica inicial da massa (pode ser vetor ou matriz)
    if(is.null(M1))  M1 <- rep(1, len = npar) # Se M1 = NULL → massa unitária → M1 = I. 
    if(!(is.vector(M1) | is.matrix(M1))) stop("Metric must be vector or matrix")
    
    adapt_mass     <- control$adapt_mass      # matriz de massa será adaptada durante o warmup?
    
    
    # [4.2] Definição das janelas de adaptação no warmup (seguindo Stan)
    w1             <- control$w1              # Janela 1: adaptação do tamanho de passo (stepsize)
    w2             <- control$w2              # Janela 2: adaptação da matriz de massa
    w3             <- control$w3              # Janela 3: ajuste fino final
    aws            <- w2                      # Tamanho da janela 2
    anw            <- w1 + w2                 # Ponto de início da janela 3
    
    # [4.3] Se o número total de warmup for menor que a soma das três janelas,
    #       não é possível adaptar a matriz de massa de forma estável → desativa
    if(warmup < (w1 + w2 + w3) && adapt_mass) 
      {
        warning("Too few warmup iterations to do mass matrix adaptation.. disabled")
        adapt_mass <- FALSE
      }
    
    # --- [6]  Aplicar a rotação no espaço de parâmetros, se M1 ≠ identidade
    
    is_identity <- (is.vector(M1) && all(M1 == 1)) || (is.matrix(M1) && all(M1 == diag(npar)))
    if(!is_identity)
    {
      rotation   <- .rotate_space(fn    = fn,
                                  gr    = gr,
                                  M     = M1,
                                  y.cur = init)
      
      # --- [7] Atualiza funções e valor inicial para o espaço transformado (x = A⁻¹·q)
      fn2        <-  rotation$fn2    # f̃(x) = f(A·x) função objetivo no novo espaço
      gr2        <-  rotation$gr2    # ∇f̃(x) = ∇f(q)·A gradiente no novo espaço
      theta.cur  <-  rotation$x.cur  # x.cur = A⁻¹·q.cur parâmetros iniciais transformadas
      chd        <-  rotation$chd    # matriz de transformação (A = L ou D)
      
      M_inv      <- rep(1, npar)     # inverso da massa (identidade no espaço rotacionado)
    } else {
             # Sem rotação quando M1 = identidade (rotação neutra)
              fn2        <- fn 
              gr2        <- gr
              theta.cur  <- init    
            }


    # --- [8] Matriz vazia para armazenar diagnósticos de cada iteração --- 
    sampler_params  <-  matrix( numeric(0),
                                nrow = iter,                              # nº total de iterações (inclui warmup e amostragem final)
                                ncol = 7,                                 # cda coluna armazena um diagnóstico específico
                                # Contém estatísticas de diagnóstico da dinâmica hamiltoniana
                                dimnames = list(NULL, c("accept_stat__",  # α médio ou proporção de aceitação
                                                        "stepsize__",     # tamanho de passo usado na simulação
                                                        "treedepth__",    # profundidade da árvore (NUTS) ou análogo; 
                                                                          # no HMC fixo é apenas informativo
                                                        "n_leapfrog__",   # nº total de passos leapfrog realizados na iteração
                                                        "divergent__",    # 1 se houve divergência (deltaH = (H - H0) > delta.max), 0 caso contrário 
                                                        "energy__",       # energia total do sistema (H = K + U)
                                                        "accepted__")))   # se a proposta foi aceita: 1 = aceita, 0 = rejeita
    

    
    
    
    
    # --- [9] Armazena as amostras da posterior no espaço dos parâmetros da rede neural --- 
    # Observação: aqui as amostras ainda estão no espaço original dos parâmetros da rede,
    # não no espaço rotacionado pela métrica.
    theta.out      <-  matrix(NA, 
                              nrow = iter,  # nº de amostras da posteriori (após warmup)
                              ncol = npar)  # nº de parâmetros da rede 

    
    # --- [10] Vetores para diagnósticos --- 
    n_leapfrog_vec <- rep(NA, len = iter)  # nº de passos leapfrog em cada iteração
    Er             <- rep(NA, len = iter)  # energia total (Hamiltoniana) em cada iteração
    accepted       <- rep(0,  len = iter)  # indicador de aceitação da proposta (1 = aceita, 0 = rejeita)

    
    
    # --- [12] Adaptação do stepsize ε via Dual Averaging (durante o warmup) p/ atingir adapt_delta ---
    if(useDA){
      # [12.1] Inicializa vetores para armazenar histórico da adaptação
      epsvec    <- Hbar       <- epsbar    <- rep(NA, length = warmup + 1)
      
      # [12.2] Encontra ε inicial estável com função específica (Hoffman & Gelman, 2014)
      #       - Se o usuário forneceu um valor inicial (control$stepsize), usa ele
      #       - Caso contrário, parte de um valor padrão (eps = 1)
      if(!is.null(init.eps))
      { eps   <-  epsvec[1]  <-  epsbar[1]  <- .find.epsilon(theta   = theta.cur, 
                                                             fn      = fn2, 
                                                             gr      = gr2, 
                                                             eps     = init.eps, 
                                                             verbose = FALSE, 
                                                             M_inv   = M_inv) 
      }else{  
        eps   <-  epsvec[1]  <-  epsbar[1]  <- .find.epsilon(theta   = theta.cur, 
                                                             fn      = fn2, 
                                                             gr      = gr2, 
                                                             eps     = 1, 
                                                             verbose = FALSE, 
                                                             M_inv   = M_inv)}
      
      # [12.3] Proteção contra NA/Inf, garante valor mínimo finito
      if(!is.finite(eps)){ eps       <- 1e-6
                           epsvec[1] <- epsbar[1]  <- eps}
      
      # [12.4] Inicializar parâmetros da adaptação dual averaging
      mu        <- log(4.005 * eps)
      #mu        <- log(10 * eps)
      Hbar[1]   <- 0
      
      
      # [12.5] Mostrar valor inicial de stepsize, se display >= 0
      if(display >= 0) cat("Initial Step size :", eps , "\n")
      
      
    } else {
              # [12.5] Se não usar dual averaging, inicializa vetores como NULL para manter consistência
              epsvec   <- Hbar    <-  epsbar    <- NULL
            }

    
    
    
    # --- [13] Marcar início do tempo da simulação (diagnóstico de performance)
    time.start  <- Sys.time()
    message('')
    message(paste('Starting NUTS at', time.start))
    
    
    
    
    
    
    last_alpha <- 0.5   # fallback conservador (ou adapt_delta, se preferir)
    
    # --- [14] Loop principal do algoritmo NUTS: executa as iterações MCMC ---
    for (m in 1:iter) 
      {
      # [14.1] Inicializa dos parâmetros 
      theta.minus   <-  theta.plus <- theta.cur                    # posições (theta) para esquerda, direita e posição atual
      theta.out[m,] <-  theta.cur                                  # Armazena a amostra inicial (no espaço rotacionado)  
      Er[m]         <-  if(m == 1) fn2(theta.cur) else Er[m - 1]   # Energia potencial inicial 
                                                                   # (reaproveita da iteração anterior se m > 1)
      r_sd          <-  sqrt(1 / M_inv)
      r.cur         <-  r.plus  <- r.minus <- rnorm(npar, 0, r_sd) # Amostra momento inicial r ~ N(0, M)
      
      
      # [14.2] Energia total inicial H0 = potencial + cinética
      H0            <- .calculate.H(theta = theta.cur,
                                    r     = r.cur,
                                    fn    = fn2,
                                    M_inv = M_inv)
      
      # [14.3] Slice sampling no log: logu = log(U) - H0
      logu          <-  log(runif(1)) - .calculate.H(theta = theta.cur,
                                                     r     = r.cur,
                                                     fn    = fn2,
                                                     M_inv = M_inv)
      
      # [14.4] Inicializa variáveis para construção da árvore binária NUTS:
      j             <-  0   # nível (profundidade) da árvore
      n             <-  1   # nº de amostras válidas na árvore
      s             <-  1   # 1 = continua expandindo
      divergent     <-  0   # indicar divergência durante a trajetória
      
      
      # [14.5] Estrutura para rastrear nº de chamadas e divergências durante a construção da árvore
      info         <-  as.environment(list(n.calls = 0, divergent = 0))
      
      # [14.6] Expande árvore até critério de parada (s == 1)
      while(s == 1){ 
        
                    # [14.9.1] Sorteia a direção da expansão: +1 = para frente, -1 = para trás
                    v   <-  sample(x = c(1, -1), size = 1)
                    
                    if(v == 1){
                        # [14.9.2] Expande árvore no sentido "direita" (forward)
                        res <- .buildtree(theta = theta.plus,    # posição inicial na ponta direita
                                          r     = r.plus,        # momento na ponta direita
                                          logu  = logu,          # limite de slice
                                          v     = v,             # direção
                                          j     = j,             # profundidade atual
                                          eps   = eps,           # tamanho de passo leapfrog
                                          H0    = H0,            # energia inicial
                                          fn    = fn2,           # função objetivo
                                          gr    = gr2,           # gradiente
                                          info  = info,          # estatísticas internas
                                          M_inv = M_inv)         # inverso da massa (identidade no espaço rotacionado)
                        
                        # [14.9.3] Atualiza extremidade direita            
                        theta.plus <- res$theta.plus
                        r.plus     <- res$r.plus
                        
                    }else{
                            # [14.9.4] Expande árvore no sentido "esquerdo" (backward)
                            res <- .buildtree(theta = theta.minus,
                                              r     = r.minus,
                                              logu  = logu,
                                              v     = v,
                                              j     = j,
                                              eps   = eps,
                                              H0    = H0,
                                              fn    = fn2,
                                              gr    = gr2,
                                              info  = info,
                                              M_inv = M_inv)
                            
                            # [14.9.5] Atualiza extremidade esquerda
                            theta.minus <- res$theta.minus
                            r.minus     <- res$r.minus
                          }
                  
                  # [14.9.6] Atualiza estado corrente se amostra for válida
                  if(!is.finite(res$s)) res$s <- 0
                  if(res$s == 1) 
                    {
                      if(runif(n = 1, min = 0, max = 1) <= res$n / n) 
                        {
                          theta.cur        <-  res$theta.prime
                          if(display >= 1){cat("New Error: ", fn2(theta.cur), "\n")}
                          
                          Er[m]            <-  fn2(theta.cur)
                          accepted[m]      <-  1
                          
                          # Transformação X → Y (reversa da rotação aplicada no início)
                          # - No espaço X, M_inv é identidade
                          # - No espaço Y, preservamos a interpretação dos parâmetros originais
                          #    - Se M1 é escalar, rotação é multiplicação simples
                          #    - Se M1 é matriz, usa-se produto matricial
                          #theta.out[m,]    <-  if(is.vector(M1)) chd * theta.cur else t(chd %*% theta.cur)
                          if(exists("chd")) {theta.out[m,] <- if(is.vector(M1)) chd * theta.cur else t(chd %*% theta.cur)  
                          } else {           theta.out[m,] <- theta.cur}       # já está no espaço original
                          
                          
                        }
                    }
                  
                  
                  # [14.9.7] Atualiza contadores e teste de U-turn
                  n                    <- n + res$n
                  s                    <- as.vector(res$s * .test.nuts(theta.plus  = theta.plus, 
                                                                       theta.minus = theta.minus, 
                                                                       r.plus      = r.plus, 
                                                                       r.minus     = r.minus))
                  if(!is.finite(s))  s <- 0
                  j                    <- j + 1
                  
                  # [14.9.8] Para se atingir profundidade máxima
                  if(j >= max_td) {
                    warning("j larger than max_treedepth, skipping to next m")
                    break}
                }
      
      
      
      # [14.10] Armazena profundidade da árvore utilizada nesta iteração
      n_leapfrog_vec[m]     <- j - 1
      
      # [14.11] Calcula taxa de aceitação média (ᾱ) desta iteração
      #         fallback neutro para folhas divergentes
      #alpha2                        <- res$alpha / res$nalpha
      #if(!is.finite(alpha2)) alpha2 <- 0
      #alpha2 <- if(res$nalpha > 0) res$alpha / res$nalpha else 0.6 # valor neutro conservador
      
      if(res$nalpha > 0){  alpha2      <-  res$alpha / res$nalpha
                           last_alpha  <-  alpha2                  # atualiza o último alpha válido
      } else {
                # Nenhuma contribuição alpha recolhida nesta iteração.
                # Use fallback conservador: reutiliza o último alpha válido (mais estável que usar adapt_delta)
                           alpha2      <-  last_alpha
      }
      
      
      
      # [14.12] Adaptação do stepsize ε via Dual Averaging (durante o warmup) p/ atingir adapt_delta
      if (useDA) 
        {
          if (m <= warmup) {
            # [14.12.1] Atualiza H̄ (estimativa acumulada do erro de aceitação)
            #            H̄[m+1] = média ponderada entre H̄[m] e (adapt_delta - ᾱ)
            Hbar[m + 1]     <- (1 - 1 / (m + t0)) * Hbar[m] + (adapt_delta - alpha2) / (m + t0)
            
            # [14.12.2] Atualiza log(ε) com base na estatística H̄ acumulada
            #           Direciona o passo ε para atingir a taxa de aceitação-alvo
            logeps         <- mu - (sqrt(m) * Hbar[m + 1] / gamma)
            
            epsvec[m + 1]  <- exp(logeps)               # valor instantâneo de ε na iteração atual do warmup,
                                                        # usado durante o warmup para gerar a próxima amostra
            epsvec[m + 1]  <- min(max(exp(logeps), 1e-6), 1e+1)
            #epsvec[m + 1]  <- max(epsvec[m + 1], 1e-6)  
            
            # [14.12.3] Suaviza log(ε) com média móvel exponencial (decaimento ∝ m^{-κ})
            #           → estabiliza a adaptação ao longo do warmup
            logepsbar      <- m ^ (-kappa) * logeps + (1 - m ^ (-kappa)) * log(epsbar[m])
            epsbar[m + 1]  <- exp(logepsbar)      
            epsbar[m + 1]  <- min(max(exp(logepsbar), 1e-6), 1e+1)
            #epsbar[m + 1]  <- max(epsbar[m + 1], 1e-6)
            
            # [14.12.4] Define ε atual para o próximo leapfrog
            eps            <- epsvec[m + 1]
            #eps <- min(max(eps, 1e-6), 1.0)
            
          } else {# [14.12.5] Após o warmup, fixa ε no valor suavizado final
            eps            <- epsbar[warmup]  # valor ε suavizado / médio ponderado ao longo do warmup
            eps            <- max(eps, 1e-6)  # ε suavizado, usado como valor fixo após o warmup para todas as amostras da posterior.
          }
        }
      
      
      
      ## [14.13] Adaptação da matriz metric M1 durante o warmup
      ##           - Executada apenas na "slow phase" da adaptação
      ##           - Cálculos feitos no espaço original (Y), mas o NUTS opera em X
      ##           → é necessário converter para Y ao atualizar M
      
      if(adapt_mass & .slow_phase(m, warmup, w1, w3)) 
        {
            ## If in slow phase, update running estimate of variances
            ## The Welford running variance calculation, see
            ## https://www.johndcook.com/blog/standard_deviation/
        
            if(m == w1){
              # [14.13.1] Inicializa o cálculo da variância acumulada (Welford) no final da 1ª janela rápida
              m1          <- theta.out[m,]        # média acumulada
              s1          <- rep(0, len = npar)   # soma acumulada dos quadrados
              k           <- 1                    # contador de amostras
              
            }else if(m == anw){
              # [14.13.2] Final de janela de adaptação → atualiza M com variância estimada
              M1          <- as.numeric(s1 / (k - 1))  # estima variância dos parâmetros
              
              # [14.13.3] Verifica valores não finitos; se houver, volta para M unitária
              if(any(!is.finite(M1)))
                {
                  warning("Non-finite estimates in mass matrix adaptation -- reverting to unit")
                  M1      <- rep(1, length(M1))
              }
              
              # [14.13.4] Recalcula funções no novo espaço rotacionado com M1 atualizada
              rotation    <- .rotate_space(fn    = fn, 
                                           gr    = gr, 
                                           M     = M1, 
                                           y.cur = theta.out[m,])
              
              fn2         <- rotation$fn2
              gr2         <- rotation$gr2
              chd         <- rotation$chd
              theta.cur   <- rotation$x.cur
              
              # [14.13.5] Reseta estatísticas acumuladas para a próxima janela
              k           <- 1
              m1          <- theta.out[m,]
              s1          <- rep(0, npar)
              
              # [14.13.6] Calcula tamanho da próxima janela de adaptação
              aws         <- 2 * aws
              anw         <- .compute_next_window(m, anw, warmup, w1, aws, w3)
            
              # [14.13.7] Busca novo ε razoável, pois M1 mudou
              eps         <- .find.epsilon(theta    = theta.cur,
                                            fn      = fn2,
                                            gr      = gr2,
                                            eps     = init.eps,
                                            verbose = FALSE,
                                            M_inv   = rep(1, npar)) # inverso da massa (identidade no espaço rotacionado)
              
              # [14.13.8] Log opcional de depuração
              if(!is.null(control$verbose))
                print(paste(m, ": new range(M1) is:", round(min(M1), 5), round(max(M1), 5), ", pars",
                            which.min(M1), which.max(M1), ", eps=", eps))
            } else {
                     
                      # [14.13.9] Passo intermediário da janela → atualiza média e soma de quadrados (Welford)
                      k   <-  k + 1
                      m0  <-  m1
                      s0  <-  s1
                      
                      m1  <-  m0 + (theta.out[m,] - m0) / k
                      s1  <-  s0 + (theta.out[m,] - m0) * (theta.out[m,] - m1)
                    }
          }
      
      
      
      
      # [14.14] Final da adaptação da matriz de massa nesta iteração
      sampler_params[m,]           <-  c(alpha2,          # taxa média de aceitação da iteração → usado para adaptação do stepsize (Dual Averaging).
                                         eps,             # tamanho de passo atual, adaptado ou fixo.
                                         j,               # profundidade da árvore NUTS nesta iteração.
                                         info$n.calls,    # nº de avaliações da função alvo durante a construção da árvore.
                                         info$divergent,  # indicador de divergência durante a trajetória.
                                         fn2(theta.cur),  # energia potencial no estado atual, já no espaço rotacionado X (coerente com todo o loop).
                                         accepted[m])     # indicador de aceitação desta amostra
      
      # [14.14.1] Registra tempo total do warmup na última iteração de adaptação
      if(m == warmup)  time.warmup <-  difftime(Sys.time(), time.start, units = 'secs')
      
      # [14.14.2] Imprime progresso do MCMC (se habilitado)
      .print.mcmc.progress(m, iter, warmup, chain, parallel, f1)
    } ## end of MCMC loop
    
    
    

    
    # --- [15] Processa os resultados finais para retorno
    
    
    theta.out       <-  cbind(theta.out, Er)  # Adiciona a coluna de energia potencial (Er) às amostras 
    theta.out       <-  theta.out[seq(1, nrow(theta.out), by = thin),] # Aplica "thinning" para reduzir autocorrelação, 
                                                                       # selecionando amostras a cada 'thin' iterações
    
    warm            <-  warmup / thin                                  # Calcula o n. de iterações de warmup após o thinning
    sampler_params  <-  sampler_params[seq(1, nrow(sampler_params), by = thin), ] # Thinning das estatísticas do sampler
    ndiv            <-  sum(sampler_params[-(1:warm), 5])              # Conta divergências ocorridas após o warmup
    
    
    # [15.6] Informa ao usuário se houve divergências pós-warmup, indicando possíveis problemas de amostragem
    if(ndiv > 0) message(paste0("There were ", ndiv, " divergent transitions after warmup"))
    
    # [15.7] Calcula e exibe a taxa média final de aceitação, ignorando warmup
    msg             <-  paste0("Final acceptance ratio=", sprintf("%.2f", mean(sampler_params[-(1:warm), 7])))
    if(useDA) msg   <-  paste0(msg, ", and target=", adapt_delta)
    message(msg)
    
    # [15.8] Mostra o tamanho final do passo ε e quantas iterações de warmup foram feitas (se DA ativo)
    if(useDA)
    {
      #message(paste0("Final step size=",round(eps, 3),"; after ",warmup," warmup iterations"))
      eps_final     <- max(epsbar[warmup], 1e-6)  # evita mostrar zero
      message(paste0("Final step size=", round(eps_final, 6),"; after ", warmup, " warmup iterations"))
    }
      
    
      
      
    # [15.9] Calcula o tempo total gasto na execução do algoritmo
    time.total      <-  difftime(Sys.time(), time.start, units = 'secs')
    
    
    # [15.10] Imprime o tempo gasto no warmup e no total da execução
    .print.mcmc.timing(time.warmup = time.warmup, time.total = time.total)
    
    return(list(par            = theta.out,       # par: amostras após thinning e com energia potencial
                sampler_params = sampler_params,  # sampler_params: estatísticas do sampler (aceitação, passo, divergências, etc.)
                time.total     = time.total,      # time.total: duração total da execução
                time.warmup    = time.warmup,     # time.warmup: duração do warmup
                warmup         = warm,            # warmup: número de iterações de warmup após thinning
                max_treedepth  = max_td))         # max_treedepth: profundidade máxima permitida da árvore NUTS
  }
