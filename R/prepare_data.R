prepare_data <- function(y, ncores = 1) {
  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1), add = TRUE)

  t1 <- y[, "time1"]
  t2 <- y[, "time2"]
  cens1 <- y[, "event1"]
  cens2 <- y[, "event2"]

  X <- as.matrix(cbind(t1, t2))
  delta <- as.matrix(cbind(cens1, cens2))

  ## Check whether first delta1=delta2=1

  row_index <- which(delta[, 1] == 1 & delta[, 2] == 1, arr.ind = TRUE)[1] # First observation with delta1=delta2=1

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

  delta.prod <- delta(delta[, 1], delta[, 2])
  delta.prod1 <- c(t(delta.prod))
  delta.prod2 <- c(delta.prod)
  rm(delta.prod)

  return(list(
    X = X,
    delta = delta,
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
