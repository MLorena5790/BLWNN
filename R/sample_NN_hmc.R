# ============================================================
# BLWNN - Bayesian Learning for Wavelet Neural Networks
#
# Function: .sample_NN_hmc
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
# Purpose: Hamiltonian Monte Carlo posterior sampling
#          for Bayesian wavelet neural networks.
#
# Contribution: Adapted and modified from the original
#               BLNN implementation to support posterior
#               sampling within the BLWNN framework,
#               including numerical safeguards,
#               trajectory control, step-size adaptation,
#               and sampler diagnostics.
#
# Modification year: 2026
#
# SPDX-License-Identifier: GPL-3.0-only
# ============================================================


#_______________________________________________________________________________________________
# .sample_NN_hmc implementa HMC com adaptação de passo ε (Dual Averaging), com:
#         - energia potencial U(q) = - log P(q|D) = βE_D(q) + αE_W(q)
#         - energia cinética  K(p) = (1/2) * pᵀ M⁻¹ p, onde p ~ N(0, M) e M é a matriz de massa
#_______________________________________________________________________________________________

.sample_NN_hmc   <-   function(iter,                 # n. de amostras extraídas da dist. posteriori 
                               fn,                   # Função auxiliar: calcula o custo regularizado 
                                                     # U --> Energia potencial do hamiltoniano
                               gr,                   # Função auxiliar: calcula o gradiente da rede 
                                                     # U' --> derivada da energia potencial 
                               init,                 # lista de vetores contendo os valores iniciais 
                                                     # dos parâmetros ou uma função -> i in 1:chains
                               warmup   = warmup,    # n. de iterações/amostras de aquecimento
                                                     # metade do n. de iterações (iter)
                               chain    = i,         # i corresponde a cadeia que será executada 
                               thin     = thin,      # taxa de diluição a aplicar às amostras
                               seed     = seeds[i],  # vetor de semente/ 01 para cada cadeia 
                               control  = control,   # lista com argumentos de controle necessários 
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
                                                     # Argumentos usados apenas para HMC
                                                     #    lambda        -> comprimento de uma trajetória = eps*L 
                                                     # param_blocks     -> índices dos blocos de parâmetros 
                               
                               display  = 0,         # ajuda a rastrear o algoritmo do amostrador exibindo vários resultados
                               path     = getwd(),   # caminho para salvar resultados, se necessário.
                               parallel = FALSE,     # valor booleano para verificar se deve executar as cadeias em núcleos paralelos ou não.
                               f1       = NULL,
                               ...) 
{

  # --- [1] Função p/ configurar parâmetros de controle do HMC/NUTS (default)  ---
  #         Define constantes físicas do sistema hamiltoniano
  .update_control <- function(control)
      { 
            # [1.1] Valores padrão para os parâmetros de controle do HMC/NUTS
            default   <-   list( #both NUTS and HMC
                                adapt_delta   = 0.65, # Taxa (%) de aceitação (HMC -> 0.65)
                                momentum.mass = 1,    # Variância do momento (massa)
                                stepsize      = NULL, # Tamanho de passo (ε)
                                useDA         = TRUE, # Dual Averaging (ajustar ε de forma adaptativa)
                                gamma         = 0.05, # Taxa de aprendizado DA (Hoffman & Gelman, 2014)
                                t0            = 10,   # Parâmetro de suavização DA (Hoffman & Gelman, 2014)
                                kappa         = 0.75, # Decaimento DA (Hoffman & Gelman, 2014)
                                metric        = NULL, # Métrica da massa 
                                adapt_mass    = FALSE,# Adaptação da métrica
                                w1            = 75,   # Janela 1 de warmup (para adaptação da massa)
                                w2            = 50,   # Janela 2
                                w3            = 25,   # Janela 3
                                max_treedepth = 10,   # Profundidade máxima da árvore NUTS — NUTS apenas
                                Lambda        = 0.25, # Comprimento da trajetória (λ = ε × L) — HMC apenas
                                param_blocks  = NULL) # índices dos blocos de parâmetros (opcional)
                                
            
            # [1.2] Se 'metric' é fornecida (matriz), desativa adapt_mass (evita conflito). 
            if(!is.null(control) && is.matrix(control$metric) && isTRUE(control$adapt_mass)) {
              warning("Adaptação de matriz de massa desativada se a métrica for uma matriz")
              control$adapt_mass <- FALSE}
        
            # [1.3] Substituindo em default os parâmetros declarados
            new <- default
            if(!is.null(control)) new <- modifyList(default, control)
            
            # [1.4] Lista final c/ parâmetros de controle HMC/NUTS
            return(new)
      }
  
  
  # --- [2] Define parâmetros de controle para HMC/NUTS ---
  control  <- .update_control(control)
  
  
  # --- [3] Define semente para reprodutibilidade HMC/NUTS ---
  #         Únicas fontes de aleatoriedade no HMC:
  #                                      - amostragem do momento p ~ N(0, M)
  #                                      - decisão de aceitação do Metropolis-Hasting
  if(!is.null(seed)) set.seed(seed)
  
  
  
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
  M1             <- control$metric
  init.eps       <- control$stepsize         
  Lambda         <- control$Lambda    
  
  init           <- as.vector(unlist(init)) # Vetor com os valores iniciais dos parâmetros da rede
  npar           <- length(init)            # N. total de parâmetros (w1, b1, w2, b2, t_tilde, log(s))
  param_blocks   <- control$param_blocks    # índices dos blocos de parâmetros 

  
  # --- [4] Definição das estruturas que armazenam resultados da simulação HMC ---
  M_inv       <- 1 / M                # Inverso da matriz de massa M -> uma matriz diagonal 
  accepted    <- rep(FALSE, iter)     # Se a proposta foi aceita em cada iteração   
  divergence  <- rep(FALSE, iter)     # Se houve divergência numérica na trajetória
  Er          <- rep(NA_real_, iter)  # Vetor de energias (usado para diagnóstico)

  
  # [4.1] Armazena as amostras da posterior no espaço dos parâmetros da rede neural
  theta.out   <- matrix(NA, 
                        nrow = iter, # n. de amostras da posteriori (após warmup)
                        ncol = npar) # n. de parâmetros da rede 
  
  
  
  # --- [5]  Se a matriz de massa M1 foi fornecida (vetor ou matriz), aplicamos a rotação ---
  if(!is.null(M1))
  {

            # [5.1] Função p/ rotacionar o espaço de parâmetros (usando M) p/ melhorar a eficiência do HMC. 
            #       Leva os parâmetros init(q) para x.cur(x) (espaço transformado). 
            #       Aplicar mudança de variáveis: q = A·x  ⇒  x = A⁻¹·q
            #       Essa rotação ajuda a padronizar a geometria do espaço, acelerando o leapfrog.
            .rotate_space  <-  function (fn, gr, M, theta.cur)
            {
              # [5.1.1] Caso 1: M é uma matriz completa (massa densa, com correlações) 
              #                 Transformação linear do espaço: x = L⁻¹·q, com M = L·Lᵗ
              if(is.matrix(M)) 
              {
                chd      <-  t(chol(M))                                      # L = t(chol(M)) matriz triangular inferior tq M = L × Lᵗ = diag(M)
                chd.inv  <-  solve(chd)                                      # L⁻¹ — usada para mudar de coordenadas q → x
                fn2      <-  function(x) fn(chd %*% x)                       # f̃(x) = f(L·x) fn2 = fn(q = L.x)
                gr2      <-  function(x) as.vector(t(chd) %*% gr(chd %*% x)) # ∇f̃(x) = ∇f(q) . L, com q = L·x
                x.cur    <-  as.numeric(chd.inv %*% theta.cur)                   # q' = x = L⁻¹·q  — coordenadas transformadas
                
              # [5.1.2] Caso 2: M é um vetor (massa diagonal: M = diag(m₁, ..., m_n))
              #                 Transformação de escala: x = D⁻¹·q, com D = diag(√M)
              #                 M como matriz de massa diagonal ⇒ sem correlações entre parâmetros
              }else if(is.vector(M))
              {
                chd    <-  sqrt(M)                                  # D = diag(√m₁, ..., √m_n)
                fn2    <-  function(x) fn(chd * x)                  # f̃(x) = f(D·x)
                gr2    <-  function(x) chd * gr(chd * x)            # ∇f̃(x) = ∇f(q) · D
                x.cur  <-  (1 / chd) * theta.cur                    # x = D⁻¹·q
              }
              
              # [5.1.3] Verificar se M é matriz ou vetor 
              else{stop("Mass matrix must be vector or matrix")}
              
              # [5.1.4] Retorna funções e parâmetros transformados no novo espaço
              return(list(gr2   = gr2,   # Gradiente transformado: ∇f̃(x)
                          fn2   = fn2,   # Função objetivo transformada: f̃(x)
                          x.cur = x.cur, # Valor inicial transformado: x = A⁻¹·q
                          chd   = chd))  # Matriz de transformação: A (L ou D)
            }
        
        
        # [5.2] Aplicar a rotação no espaço de parâmetros 
        rotation <- .rotate_space(fn        = fn,
                                  gr        = gr,
                                  M         = M1,
                                  theta.cur = init)
        
        # [5.3] Atualiza funções e valor inicial para o espaço transformado (x = A⁻¹·q)
        fn2        <-  rotation$fn2    # f̃(x) = f(A·x) função objetivo no novo espaço
        gr2        <-  rotation$gr2    # ∇f̃(x) = ∇f(q)·A gradiente no novo espaço
        current.q  <-  rotation$x.cur  # x.cur = A⁻¹·q.cur parâmetros iniciais transformadas
        chd        <-  rotation$chd    # matriz de transformação (A = L ou D)
   
    # --- [6] Se M1 = NULL, não aplicamos a rotação, permanece no espaço original q --- 
  } else {
            fn2        <- fn; 
            gr2        <- gr
            current.q  <- init       
          }
    
  
  
  
  
  # --- [7] Matriz vazia para armazenar diagnósticos de cada iteração/ amostras extraídas da posteriori --- 
  sampler_params  <-  matrix( numeric(0),
                              nrow = iter,                              # iter = n. de amostras 
                              ncol = 5,
                              # Contém estatísticas de diagnóstico da dinâmica hamiltoniana
                              dimnames = list(NULL, c("accept_stat__",  # prob. de aceitação da proposta
                                                      "stepsize__",     # tamanho de passo usado na simulação
                                                      "int_time__",     # tempo de integração (step_size × número de passos)
                                                      "energy__",       # energia do sistema (H = K + U)
                                                      "accepted__")))   # se a proposta foi aceita: 1 = aceita, 0 = rejeita
  
  

  
  # --- [8] Função para encontrar um valor razoável de ε (stepsize) inicial p/ DA. --- 
  #         Estratégia: dobra ou divide ε até que a aceitação seja > 0.5
  #         Baseado em Hoffman & Gelman (2014, p. 1608)
  .find.epsilon <- function( theta,          # valor atual dos parâmetros: θ
                             fn,             # energia potencial: U(θ) = -log π(θ)
                             gr,             # gradiente: ∇U(θ)
                             eps     = 0.1,    # valor inicial de stepsize
                             verbose = TRUE,
                             M_inv)          # massa inversa (vetor: assume diagonal)
  {
    
    # [8.1] Amostragem do momentum inicial p ~ N(0, M^{-1}) 
    #       Fixo durante o ajuste de ε
    r <- rnorm(n = length(theta), mean = 0, sd = sqrt(1/M_inv))
    
    
    # [8.2] Função auxiliar: calcula o Hamiltoniano H(θ, p) = U(θ) + K(p), 
    #                        onde U(θ) = fn(θ) e K(p) = (1/2) * pᵗ M⁻¹ p
    .hamiltonian <-  function(theta, r, fn, M_inv) 
    {
      fn(theta) + (1 / 2) * sum(M_inv * r ^ 2) 
    }
    
    # [8.3] Um passo leapfrog com ε atual
    r.new     <- r + (eps / 2) * gr(theta)          # Meio passo no momentum
    theta.new <- theta + eps * (M_inv * r.new)      # Passo completo na posição
    r.new     <- r.new + (eps / 2) * gr(theta.new)  # Meio passo final 
    
    
    # [8.4] Calcular a energia antes e depois do leapfrog
    H1        <- .hamiltonian(theta = theta, r = r, fn = fn, M_inv)        # Fixo durante o ajuste de ε   
    H2        <- .hamiltonian(theta = theta.new, r = r.new, fn = fn, M_inv)
    
    
    # [8.5] Qual é a probabilide de aceitação da proposta do leapfrog? 
    #       Queremos aceitação ≈ 0.5. 
    #       Se > 0.5 → a = 1, se <  0.5 → a = -1. (determina a direção do ajuste)
    accept_prob          <- exp(min(0, H1 - H2))             # evita valores > 1
    a                    <- ifelse(accept_prob > 0.5, 1, -1) # a é fixo por construção teórica
    if(!is.finite(a)) a  <- -1
  
    
   
    # [8.6] Ajustar ε até que ΔH = H2 - H1 aceitável: a*ΔH < -a*log(2)
    #       Se accept_prob > 0.5, então a = 1 e o critério vira ΔH < log(2) → Dobra eps
    #       Se accept_prob < 0.5, então a = -1 e o critério vira -ΔH < log(2) → Dividir eps
    #       Se accept_prob ≈ 0.5, critério satisfeiro a*ΔH < -a*log(2), parar!
    k  <- 1  # contador de iterações
    
    # O loop continua enquanto:
    #                         - !is.finite(H1)             : leapfrog explodiu → ε mto grande
    #                         - !is.finite(H2)             : leapfrog explodiu → ε mto grande
    #                         - a * (H2 - H1) > -a * log(2): aceitação acima ou abaixo de 0.5
    while(!is.finite(H1) | !is.finite(H2) | a * (H2 - H1) > -a * log(2)) 
    {
      
      # [8.7] Ajusta ε: dobra ou divide por 2
      #               - a = 1  → eps = eps*2
      #               - a = -1 → eps = eps*0.5      
      eps       <- eps * ifelse(a == 1, 2, 0.5)  
      
      
      # [8.8] Reexecutar o passo leapfrog com novo ε
      r.new     <- r + (eps / 2) * gr(theta)
      theta.new <- theta + eps * (M_inv * r.new)   
      r.new     <- r.new + (eps / 2) * gr(theta.new)
      
      
      # [8.9] Recalcular a energia
      H2        <- .hamiltonian(theta = theta.new, r = r.new, fn = fn, M_inv)
      k         <- k + 1
      
      
      # [8.10] EEvita looping infinito — limite de 50 iterações
      if(k > 50) 
      {stop("Não foi possível encontrar um epsilon razoável após 50 tentativas. 
             Verifique a especificação do modelo ou da massa.")}
    }
    
    # [8.11] Exibe o valor encontrado, se verbose = TRUE
    if(verbose)
      message(paste("Epsilon  razoável=", eps, "encontrado depois de", k, "passos"))
    
    # [8.12] Retorna ε, de forma invisível (verbose = FALSE)
    return(invisible(eps))
  }
  
  
  
  # --- [9] Inicialização do stepsize (ε) para adaptação via Dual Averaging  --- 
  # [9.1] Vetores para armazenar ε e estatísticas durante o warmup (aquecimento)
  epsvec  <-  Hbar      <-  epsbar     <-  rep(NA, length = warmup + 1)
  
  # [9.2] Estimar valor inicial de epsilon com base na função .find.epsilon()
  #       - Se o usuário forneceu um valor inicial (control$stepsize), usa ele
  #       - Caso contrário, parte de um valor padrão (eps = 1)
  if(!is.null(init.eps))
  { eps   <-  epsvec[1]  <-  epsbar[1]  <- .find.epsilon(theta = init, fn, gr, eps = init.eps, verbose = FALSE, M_inv) 
  }else{  
    eps   <-  epsvec[1]  <-  epsbar[1]  <- .find.epsilon(theta = init, fn, gr, eps = 1, verbose = FALSE, M_inv)}
  
  # [9.3] Mostrar valor inicial de stepsize, se display >= 0
  if(display >= 0) cat("Initial Step size :", eps , "\n")
  
  # [9.4] Inicializar parâmetros da adaptação dual averaging
  #mu      <- log(4.005 * eps)  # alvo de log-epsilon (valor médio ao final da adaptação)
  mu      <- log(10 * eps)
  Hbar[1] <- 0                 # valor acumulado de erro de aceitação

  # [9.5] Marcar início do tempo da simulação (diagnóstico de performance)
  time.start  <- Sys.time()
  message('')
  message(paste('Starting HMC', time.start))
  
  
  
  
  
  
  
  # --- [10] Loop principal do algoritmo HMC (iterações MCMC)  --- 
  for(m in 1:iter) 
  {
    start_iter <- Sys.time()
    
    cat("Entrando na iteração", m, "\n")
    
    
    # [10.1] Número de passos (L) da trajetória Leapfrog, baseado em Lambda e ε (mínimo 1)
    L   <- min(max(3, round(Lambda / eps)), 200)
    #L  <-  max(1, round(Lambda / eps))
    if(display >= 2) cat("Number of Leaps :", L , "\n")
    
    
    # [10.2] Amostra momentos iniciais p ~ N(0, M)
    #        Cada parâmetro da cadeia (q) tem um momento associado (p). 
    #        São sorteados da normal padrão escalada pela massa M.
    p.cur      <- rnorm(length(current.q), 0, sqrt(M)) 
    p.new      <- p.cur 
  
    
    
    # [10.3] Inicializa posição e gradiente (current.q está no espaço rotacionado)
    q.new      <- current.q      # nova posição começa na atual q 
    current.g  <- gr2(current.q) # gradiente atual ∇U(q)
    new.g      <- current.g      # novo gradiente, começando igual (atualiza depois)
    
    
    # [10.4] Energia total atual: H = U(q) + K(p)
    current.K  <- sum(M_inv * p.cur ^ 2) / 2 # K(p): energia cinética atual (quadrática) 
    current.H  <- fn2(current.q) + current.K # Hamiltoniano atual: H(q,p) = U[q] + K(p) 

 
    # [10.5] Aleatoriza ε após o warmup (se DA ativo) para evitar trajetórias periódicas (opcional) 
    #        Evita que o sistema caia em trajetórias com periodicidade (~2π) constante.
    #        Resolve, também, a questão do limite de estabilidade para várias regiões do espaço de estado. 
    if(useDA & m > warmup) eps <- eps * runif(n   = 1, min = 0.9, max = 1.1) 
    
    # --- Sempre recalcula L logo depois ---
    eps <- min(max(eps, 1e-4), 1.0)
    L   <- min(max(3, round(Lambda / eps)), 200)

    
    # [10.6] Meio passo inicial do Leapfrog
    p.new <- p.new - eps * current.g/ 2      
    q.new <- q.new + eps * (M_inv * p.new)             
    new.g <- gr2(q.new)                      
    
    
    # [10.7] Passos internos do Leapfrog
    for(i in 1:(L-1)) 
    {
      p.new <- p.new - eps * new.g 
      q.new <- q.new + eps * (M_inv * p.new)
      new.g <- gr2(q.new)

      # [10.7.1] Teste de segurança: aborta se ocorrerem valores infinitos/NaN
      if (any(!is.finite(p.new)) | any(!is.finite(q.new))) break
    }
    
    # [10.8] Meio passo final no momento
    p.new <- p.new - eps * new.g / 2
    
 
    
    # [10.9] Calcula energia da proposta H = U(q) + K(p)
    proposed.U <- fn2(q.new)                 # U(q): potencial da nova posição
    proposed.K <- sum(M_inv * p.new ^ 2) / 2 # K(p): energia cinética da nova proposta
    proposed.H <- proposed.U + proposed.K    # H(q,p): Hamiltoniano da proposta
    acceptProb <- (current.H - proposed.H)
    
    if(display == 3) cat("Iteration :",m, "---", "(Old Eng, New Eng)", " (",current.H,proposed.H,")", "\n")
                    
    
    
    
    # [10.10] Detecta divergência se energia inválida (NaN, Inf)
    #         Quando o Hamiltoniano proposto é indefinido (ex: NaN ou Inf), a proposta é rejeitada 
    #         e marcada como divergente, para que o adaptador (ex.: dual averaging) possa responder, 
    #         normalmente reduzindo ε.
    if(!is.finite(acceptProb)) 
    {
          divergence[m]  <-  1     # registrando divergência 
          acceptProb     <-  -Inf  # garantir que log(runif(1)) < acceptProb nunca ocorra.
      
    }else{divergence[m]  <-  0}
    
    
    # [10.11] Critério de aceitação de Metropolis: 
    #         Sorteia u ~ U(0,1); aceita q.new se log(u) < ΔH = H_antiga - H_nova.
    #         Caso contrário, rejeita e mantém current.q.
    if(is.finite(acceptProb) && log(runif(1)) < acceptProb) 
    {
          current.q     <-  q.new
          accepted[m]   <-  TRUE
          if(display >=1) cat("New potential energy:" , fn2(current.q), "\n")
          
    }else{accepted[m]   <-  FALSE}
  
    
    
    # [10.12] Armazena o estado atual da cadeia e energia atual:
    #         - theta.out salva o valor de q (parâmetros) na iteração m
    #         - Er registra a energia associada ao ponto atual (log da densidade a posteriori) 
    #           U(q) = −log P(q|D)  →  log P(q|D) = −U(q)
    theta.out[m,]  <- current.q
    Er[m]          <- fn2(current.q)

    
    
    # [10.13] Adaptação do stepsize ε via Dual Averaging (durante o warmup) p/ atingir adapt_delta
    if(useDA) 
    {
      if(m <= warmup)
      {
        # [10.13.1] Atualiza a estatística acumulada H̄ usada para controlar a taxa de aceitação
        #           H̄[m+1] ← média ponderada entre H̄[m] e o erro de aceitação na iteração m
        Hbar[m + 1]   <-  (1 - 1 / (m + t0))*Hbar[m] + 
                          (adapt_delta - min(1, exp(acceptProb))) / (m + t0)
        
        # [10.13.2] Atualiza log(ε) com base na estatística H̄ acumulada
        #           Direciona o passo ε para atingir a taxa de aceitação-alvo
        logeps        <-  mu - sqrt(m)*Hbar[m + 1] / gamma
        #           mu define o valor-alvo de log(epsilon) ao final da adaptação.
        #           Esse alvo serve como "centro" para onde a média móvel log(ε) será puxada.
        
        if(!is.finite(logeps)) {
                                logeps <- log(1e-3)
                                message("logeps não finito, redefinido para log(1e-3)")
                              }
        
        
        epsvec[m + 1] <-  exp(logeps)
        epsvec[m + 1] <-  max(epsvec[m + 1], 1e-4)
        
        # [10.13.3] Suaviza log(ε) com média móvel exponencial (decaimento ∝ m^{-κ})
        #           Ajuda a estabilizar a trajetória do passo durante o warmup
        logepsbar     <-  m^(-kappa)*logeps + (1 - m^(-kappa))*log(epsbar[m])
        epsbar[m + 1] <-  exp(logepsbar)
        epsbar[m + 1] <-  max(epsbar[m + 1], 1e-4)
        
        # [10.13.4] Atualiza ε com o novo valor adaptado
        eps           <-  epsvec[m + 1]
        
        # >>> Adicione aqui: garante que eps > 1e-4 e ajusta L
        eps <- min(max(eps, 1e-4), 1.0)
        L   <- min(max(3, round(Lambda / eps)), 200)
        
        if(display >= 2) cat("\n", "step size during adapt :", eps, "\n")
        
        # [10.13.5] Após o warmup, fixa ε no valor suavizado final
      } else { eps    <-  epsbar[warmup]
              # >>> Também aqui: garante que eps > 1e-4 e ajusta L
              eps <- min(max(eps, 1e-5), 1.0)
              L   <- min(max(1, round(Lambda / eps)), 200)
              } 
               
    }

    
    # [10.14] Armazena parâmetros diagnósticos da iteração m 
    sampler_params[m,] <- c(min(1, exp(acceptProb)), # taxa de aceitação
                            eps,                     # passo (stepsize) atual
                            eps * L,                 # comprimento da trajetória (ε × L)
                            fn2(current.q),          # U(q) -> energia potencial 
                            accepted[m])             # se a proposta foi aceita (TRUE/FALSE)
    
    
    # [10.15] Registra duração do warmup (somente na última iteração do warmup)
    if(m == warmup)
      time.warmup <- difftime(Sys.time(), time.start, units = 'secs')
    
 
    
    # [10.16] Função auxiliar para exibir (ou salvar) o progresso da amostragem MCMC
    .print.mcmc.progress <- function (iteration,        # número da iteração atual
                                      iter,             # total de iterações
                                      warmup,           # número de iterações de warmup
                                      chain,            # número da cadeia atual
                                      parallel = FALSE, # se a execução é paralela
                                      f1       = NULL,  # arquivo para saída paralela
                                      refresh = NULL)
    {
      i         <- iteration
      
      # [10.16.1] Define intervalo de atualização da mensagem (mínimo de 10 iterações)
      if(is.null(refresh)) refresh  <-  max(10, floor(iter / 10))
      

      
      # [10.16.1] Exibe/salva progresso se: início, fim ou múltiplo de 'refresh'
      if(i == 1|i == iter|i %% refresh == 0) 
      {
        # [10.16.3] Formata número da iteração com largura alinhada
        i.width <- formatC(i, width = nchar(iter))
        
        # [10.16.4] Monta a mensagem com porcentagem e estágio (warmup ou sampling)
        out     <- paste0("Chain ", chain, 
                          ", Iteration: ", i.width,"/", iter, 
                          " [", formatC(floor(100 * (i / iter)), width = 3),"%]",
                          ifelse(i <= warmup, " (Warmup)", " (Sampling)"), 
                          "\n")
        
        # [10.16.5] Imprime no console (modo sequencial) ou grava em arquivo (modo paralelo)
        if(!parallel)
        {message(out)}else{cat(out, file = f1, append = TRUE)}
      }
      
    }
    
    ##########################################################################
    
    # [10.17]  Exibe progresso da iteração atual
    .print.mcmc.progress(m, iter, warmup, chain, parallel, f1, refresh = 1)
    ## fim do loop MCMC
  }
  
  
  
  
  
  
  
  # --- [11] Aplica transformação reversa (espaço original) se a métrica M1 foi usada
  if(!is.null(M1)) 
  {
    # [11.1] desfaz rotação por chd (Cholesky de M1)
    theta.out  <- t(apply(theta.out, 1, function(x) chd %*% x))
  }
  
  # --- [12] Adiciona o vetor de erros Er (função de custo) às amostras
  theta.out       <- cbind(theta.out, Er)
  
  # --- [13] Aplica thining (subamostragem) nas amostras da cadeia
  theta.out       <- theta.out[seq(1, nrow(theta.out), by=thin),]
  sampler_params  <- sampler_params[seq(1, nrow(sampler_params), by=thin),]
  
  
  # --- [14] Alerta se ocorreram transições divergentes após o warmup
  if(sum(divergence[-(1:warmup)]) > 0)
    message(paste0("Houve ",
                   sum(divergence[-(1:warmup)]),
                   " transições divergentes após o aquecimento"))
  
  # --- [15] Exibe taxa de aceitação final comparada ao alvo 'adapt_delta'
  message(paste0("Taxa de aceitação final: ",
                 sprintf("%.2f", mean(accepted[-(1:warmup)])),
                 " | Taxa de aceitação alvo (adapt_delta): ",
                 adapt_delta))
  
  # --- [16] Exibe tamanho final do passo adaptado via Dual Averaging (após o warmup)
  if(useDA)
    message(paste0("Tamanho do passo final =",
                   #round(epsbar[warmup], 3),    # Último valor adaptado do passo
                   format(epsbar[warmup], scientific = TRUE),
                   "; depois ",
                   warmup,
                   "iterações de aquecimento"))
  
  # --- [17] Calcular o tempo gasto total no algoritmo HMC
  time.total      <-  difftime(Sys.time(), 
                               time.start, 
                               units = 'secs')
  
  
  
  # --- [18] Função para imprimir o tempo gasto nas fases do HMC ---
  .print.mcmc.timing  <- function (time.warmup, 
                                   time.total)
  {
    x <- " Tempo decorrido: "
    
    # [18.1] Tempo gasto apenas no aquecimento (warmup)
    message(paste0(x, sprintf("%.1f", time.warmup), " segundos (aquecimento)"))
    
    # [18.2]  Tempo gasto na fase de amostragem (tempo total - tempo warmup)
    message(paste0( x,
                    sprintf("%.1f", time.total - time.warmup)," segundos (amostragem)"))
    
    # [18.3] Tempo total do algoritmo (warmup + amostragem)
    message(paste0(x, sprintf("%.1f", time.total), " segundos (Total)"))
  }
  
  # [18.4] Chamada da função para mostrar os tempos calculados durante a execução do HMC
  .print.mcmc.timing(time.warmup = time.warmup, 
                     time.total  = time.total)
  
  
  
  
  return(fit <- list(par            = theta.out,      # Matriz das amostras geradas pelo HMC (parâmetros do modelo)
                     sampler_params = sampler_params, # Matriz com informações diagnósticas das amostras (aceitação, passo, energia, etc)
                     time.total     = time.total,     # Tempo total decorrido na execução do algoritmo HMC (warmup + sampling), em segundos
                     time.warmup    = time.warmup,    # Tempo gasto somente na fase de aquecimento (warmup), em segundos
                     warmup         = warmup / thin)) # Número efetivo de iterações descartas no warmup, após subamostragem (thinning)
  
  
}
