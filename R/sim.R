SimData <- function(
  K,
  cens.par = 0,
  alpha = c(3, 5, 1.5),
  weights = c(0.2, 0.4, 0.4),
  margin = "exp",
  ncores = 1,
  ...
) {
  # u1 <- runif(K, 0, 1)
  # u2 <- runif(K, 0, 1)
  #
  # a <- alpha^u1 + (alpha - alpha^u1)*u2
  #
  # # Fan 2000
  # T1 <- -log(u1)
  # T2 <- -log(log(a/(a+(1-alpha)*u2),base = alpha))

  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1))

  mx <- copula::mixCopula(
    list(
      copula::claytonCopula(alpha[1], dim = 2),
      copula::frankCopula(alpha[2], dim = 2),
      copula::gumbelCopula(alpha[3], dim = 2)
    ),
    w = weights
  )

  U <- copula::rCopula(K, mx)

  if (margin == "exp") {
    T1 <- -log(U[, 1])
    T2 <- -log(U[, 2])
  } else if (margin == "unif") {
    T1 <- 5 * (1 - U[, 1])
    T2 <- 5 * (1 - U[, 2])
  }

  # margin_dist <- paste0("q", margin)

  # T1 <- get(margin_dist)(p = U[,1], lower.tail = FALSE, ...)
  # T2 <- get(margin_dist)(p = U[,2], lower.tail = FALSE, ...)

  if (cens.par > 0) {
    C1 <- rexp(K, cens.par)
    C2 <- rexp(K, cens.par)

    X1 <- pmin(T1, C1)
    X2 <- pmin(T2, C2)

    X <- as.matrix(cbind(X1, X2))

    delta1 <- 1 * (T1 <= C1)
    delta2 <- 1 * (T2 <= C2)
  } else {
    X1 <- T1
    X2 <- T2

    X <- as.matrix(cbind(X1, X2))

    delta1 <- delta2 <- rep(1, K)
  }

  delta <- as.matrix(cbind(delta1, delta2))

  # # NOTE max+1 in geval van unif.ub = 5 geeft gradient=0 voor redelijk veel betas.
  # if (!is.null(unif.ub) && unif.ub < 5) {
  #   qq1 <- quantile(X1[delta1 == 1], probs = seq(0,1,length = df - degree + 2))
  #   knots1 <- c(min(X1)-1, qq1[-c(1,length(qq1))], max(X1)+1)
  #   qq2 <- quantile(X2[delta2 == 1], probs = seq(0,1,length = df - degree + 2))
  #   knots2 <- c(min(X2)-1, qq1[-c(1,length(qq2))], max(X2)+1)
  #
  #   # knots1 <- seq(min(X1)-1, max(X1)+1, length.out = df - degree + 2)
  #   # knots2 <- seq(min(X2)-1, max(X2)+1, length.out = df - degree + 2)
  # } else {
  #   qq1 <- quantile(X1[delta1 == 1], probs = seq(0,1,length = df - degree + 2))
  #   knots1 <- c(min(X1)-1, qq1[-c(1,length(qq1))], max(X1))
  #   qq2 <- quantile(X2[delta2 == 1], probs = seq(0,1,length = df - degree + 2))
  #   knots2 <- c(min(X2)-1, qq1[-c(1,length(qq2))], max(X2))
  #
  #   # knots1 <- seq(min(X1)-1, max(X1), length.out = df - degree + 2)
  #   # knots2 <- seq(min(X2)-1, max(X2), length.out = df - degree + 2)
  # }

  ## Check whether first delta1=delta2=1

  row_index <- which(delta1 == 1 & delta2 == 1, arr.ind = TRUE)[1] # First observation with delta1=delta2=1

  # Switch rows
  if (row_index > 1) {
    tmp_row_X <- X[1, ]
    tmp_row_delta <- delta[1, ]

    X[1, ] <- X[row_index, ]
    delta[1, ] <- delta[row_index, ]

    X[row_index, ] <- tmp_row_X
    delta[row_index, ] <- tmp_row_delta

    rm(tmp_row_delta, tmp_row_X, row_index)
  } else {
    X <- X
    delta <- delta
  }

  ## Calculating the risk set

  # N <- outer(X[,1], X[,2], function(x,y) mapply(riskset,x,y))
  # N1 <- c(t(N))
  # N2 <- c(N)

  N <- riskset_fast(X[, 1], X[, 2])
  mode(N) <- "integer"
  N1 <- c(t(N))
  N2 <- c(N)
  rm(N)

  # Row index of positive elements in riskset
  idxN1 <- which(N1 > 0)
  idxN2 <- which(N2 > 0)

  ## Calculating indicator functions in likelihood

  #### I(X1j >= X1i)
  # I1 <- sapply(X[,1], function(x) 1*(X[,1] >= x)) # col=1,...,i,...,n row=1,...,j,...,n
  I1 <- indgreater(X[, 1])
  mode(I1) <- "integer"

  #### I(X2j <= X2i)
  # I2 <- sapply(X[,2], function(x) 1*(X[,2] <= x)) # col=1,...,i,...,n row=1,...,j,...,n
  I2 <- indless(X[, 2])
  mode(I2) <- "integer"

  #### I(X2j >= X2i)
  # I3 <- sapply(X[,2], function(x) 1*(X[,2] >= x)) # col=1,...,i,...,n row=1,...,j,...,n
  # I3 <- t(I2); mode(I3) <- "integer"
  #
  # #### I(X1j <= X1i)
  # # I4 <- sapply(X[,1], function(x) 1*(X[,1] <= x)) # col=1,...,i,...,n row=1,...,j,...,n
  # I4 <- t(I1); mode(I4) <- "integer"

  #### I(X1j = X1i) NOTE THAT THIS IS DIAG(1,500,500) IF NO TIES
  # I5 <- sapply(X[,1], function(x) 1*(X[,1] == x)) # col=1,...,i,...,n row=1,...,j,...,n
  I5 <- indequal(X[, 1])
  mode(I5) <- "integer"

  #### I(X2j = X2i) NOTE THAT THIS IS DIAG(1,500,500) IF NO TIES
  # I6 <- sapply(X[,2], function(x) 1*(X[,2] == x)) # col=1,...,i,...,n row=1,...,j,...,n
  I6 <- indequal(X[, 2])
  mode(I6) <- "integer"

  #I1 <- lapply(X1, function(x) 1*(X2 >= x))
  #test <- matrix(unlist(I1), ncol = 500, byrow = FALSE)

  # A1 <- c(I1*outer(delta[,2], delta[,1]))[N1 > 0]
  # A2 <- c(I3*outer(delta[,1], delta[,2]))[N2 > 0]

  delta.prod = delta(delta[, 1], delta[, 2])
  delta.prod1 <- c(t(delta.prod))
  delta.prod2 <- c(delta.prod)
  rm(delta.prod)

  RcppParallel::setThreadOptions(numThreads = 1)

  return(list(
    X = X,
    idxN1 = idxN1 - 1, # C++ indexing (starts at 0)
    idxN2 = idxN2 - 1, # C++ indexing (starts at 0)
    riskset1 = N1[idxN1],
    riskset2 = N2[idxN2],
    I1 = c(I1)[idxN1],
    I2 = c(I2)[idxN1],
    I3 = c(t(I2))[idxN2],
    I4 = c(t(I1))[idxN2],
    I5 = c(I5)[idxN1],
    I6 = c(I6)[idxN2],
    delta1 = delta.prod1[idxN1],
    delta2 = delta.prod2[idxN2]
  ))
}
