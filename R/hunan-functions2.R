set_blas_threads <- function(nthreads = parallel::detectCores()) {
  blas_name <- extSoftVersion()["BLAS"]

  if (grepl("openblas", blas_name)) {
    Sys.setenv(OPENBLAS_NUM_THREADS = nthreads)
  } else if (grepl("mkl", blas_name)) {
    Sys.setenv(MKL_NUM_THREADS = nthreads)
  } else if (grepl("accelerate", blas_name) || grepl("vecLib", blas_name)) {
    Sys.setenv(VECLIB_MAXIMUM_THREADS = nthreads)
  } else {
    message("Unknown BLAS: thread setting may not take effect")
  }

  # Optionally, verify
  message("Using BLAS: ", R.version$BLAS, " with ", nthreads, " threads")
}

theta.frank <- function(x, y, alpha = 0.0023) {
  A <- (alpha - 1) * log(alpha) * alpha^(2 - exp(-x) - exp(-y))
  B <- (alpha^(1 - exp(-x)) - alpha) * (alpha^(1 - exp(-y)) - alpha)
  C <- -1 +
    exp(-x) +
    exp(-y) +
    log(
      1 + (alpha^(1 - exp(-x)) - 1) * (alpha^(1 - exp(-y)) - 1) / (alpha - 1),
      base = alpha
    )
  return(A * C / B)
  # u <- cbind(punif(x, 0, 5), punif(y,0,5))
  # s <- pCopula(u, frankCopula(param = alpha))
  # CRF <- s*alpha/(1-exp(-alpha*s))
  # return(CRF)
}

theta.mix <- function(
  t1,
  t2,
  w = c(0.2, 0.4, 0.4),
  alpha = c(3, 5, 1.5),
  margin = "unif"
) {
  if (margin == "exp") {
    S1 <- exp(-t1)
    S2 <- exp(-t2)
  } else if (margin == "unif") {
    S1 <- 1 - t1 / 5
    S2 <- 1 - t2 / 5
  }

  mx <- copula::mixCopula(
    list(
      copula::claytonCopula(alpha[1], dim = 2),
      copula::frankCopula(alpha[2], dim = 2),
      copula::gumbelCopula(alpha[3], dim = 2)
    ),
    w = w
  )

  C00 <- copula::pCopula(cbind(S1, S2), mx)
  C11 <- copula::dCopula(cbind(S1, S2), mx)
  C10 <- copula::cCopula(cbind(S1, S2), mx)[, 2]
  C01 <- copula::cCopula(cbind(S2, S1), mx)[, 2]

  CRF <- C00 * C11 / (C01 * C10)

  return(CRF)
}


WoodSpline <- function(
  t,
  delta,
  dim,
  degree = 3,
  type = "ps",
  quantile = FALSE,
  scale = TRUE,
  repara = FALSE,
  observed.region = FALSE,
  m2 = degree - 1,
  knot.margin = 0.001
) {
  # Create knot sequence for spline ----
  nk <- dim - degree + 1 # Number of "interior" knots (internal + boundary)

  # Make sure knot placement is within observed region
  xl <- min(t)
  xu <- max(t)
  xr <- xu - xl
  xl <- xl - xr * knot.margin
  xu <- xu + xr * knot.margin
  dx <- (xu - xl) / (nk - 1)
  knots <- seq(xl - dx * degree, xu + dx * degree, length = nk + 2 * degree) # Vector of knots
  if (quantile) {
    k.int <- quantile(
      ifelse(observed.region, t[delta == 1], t),
      probs = seq(0, 1, length = nk)
    )[-c(1, nk)]
    knots[(degree + 2):(length(knots) - (degree + 1))] <- k.int
  }

  X <- splines::splineDesign(knots, t, degree + 1)

  # Create penalty matrix S = t(D1) %*% D1 if necessary ----
  if (type == "bs") {
    ## Integrated squared derivative penalty ----

    pord <- degree - m2
    k0 <- knots[(degree + 1):(degree + nk)]
    h <- diff(k0)
    h1 <- rep(h / pord, each = pord)
    k1 <- cumsum(c(k0[1], h1))

    D <- splines::splineDesign(knots, k1, derivs = m2)

    P <- solve(matrix(
      rep(seq(-1, 1, length = pord + 1), pord + 1)^rep(0:pord, each = pord + 1),
      pord + 1,
      pord + 1
    ))
    i1 <- rep(1:(pord + 1), pord + 1) + rep(1:(pord + 1), each = pord + 1) ## i + j
    H <- matrix((1 + (-1)^(i1 - 2)) / (i1 - 1), pord + 1, pord + 1)
    W1 <- t(P) %*% H %*% P
    h <- h / 2 ## because we map integration interval to to [-1,1] for maximum stability
    ## Create the non-zero diagonals of the W matrix...
    ld0 <- rep(mgcv::sdiag(W1), length(h)) * rep(h, each = pord + 1)
    i1 <- c(
      rep(1:pord, length(h)) + rep(0:(length(h) - 1) * (pord + 1), each = pord),
      length(ld0)
    )
    ld <- ld0[i1] ## extract elements for leading diagonal
    i0 <- 1:(length(h) - 1) * pord + 1
    i2 <- 1:(length(h) - 1) * (pord + 1)
    ld[i0] <- ld[i0] + ld0[i2] ## add on extra parts for overlap
    B <- matrix(0, pord + 1, length(ld))
    B[1, ] <- ld
    for (k in 1:pord) {
      ## create the other diagonals...
      diwk <- mgcv::sdiag(W1, k) ## kth diagonal of W1
      ind <- 1:(length(ld) - k)
      B[k + 1, ind] <- (rep(h, each = pord) *
        rep(c(diwk, rep(0, k - 1)), length(h)))[ind]
    }
    ## ... now B contains the non-zero diagonals of W
    B <- mgcv::bandchol(B) ## the banded cholesky factor.
    ## Pre-Multiply D by the Cholesky factor...
    D1 <- B[1, ] * D
    for (k in 1:pord) {
      ind <- 1:(nrow(D) - k)
      D1[ind, ] <- D1[ind, ] + B[k + 1, ind] * D[ind + k, ]
    }
    S <- crossprod(D1)
  } else if (type == "ps") {
    ## Discrete penalty ----
    D1 <- diff(diag(dim), differences = m2)
    S <- crossprod(D1)
  } else if (type == "gps") {
    M1 <- M2 <- c()
    M <- diff(diag(dim))

    # W1
    # M1 <- diff(knots[2:(length(knots) - 1)], lag = 3) #Alternative way to compute M1 and M2
    for (i in 1:(dim - 1)) {
      M1[i] <- knots[degree + 1 + i] - knots[i + 1]
    }
    # W2
    # M2 <- diff(knots[3:(length(knots) - 2)], lag = 2)
    for (i in 1:(dim - 2)) {
      M2[i] <- knots[degree + 1 + i] - knots[i + 2]
    }

    W1 <- diag(M1) / (degree + 1 - 1)
    W2 <- diag(M2) / (degree + 1 - 2)

    D1 <- solve(W2) %*% diff(diag(dim - 1)) %*% solve(W1) %*% diff(diag(dim))
    S <- crossprod(D1)
  }

  # Scaling the penalty matrix S ----
  if (scale) {
    # maXX <- norm(X, type = "I")^2
    # maS <- norm(S) / maXX
    maXX <- norm(crossprod(X), type = 'F')
    maS <- norm(S, type = "F") / maXX
    S <- S / maS
    D1 <- D1 / sqrt(maS)
  } else {
    maS <- NULL
  }

  # Reparametrization
  if (repara) {
    # G <- t(splines::splineDesign(knots, seq(min(t),max(t),length=dim), degree+1))
    # Gm <- solve(G)
    # X <- X %*% Gm
    # S <-  t(Gm) %*% S %*% Gm
    sv <- svd(splines::splineDesign(
      knots,
      seq(min(t), max(t), length = dim),
      degree + 1
    ))
    if (sv$d[dim] / sv$d[1] < .Machine$double.eps^.66) {
      warning("Reparametrization unstable. Original model matrix returned")
      XP <- NULL
    } else {
      XP <- sv$v %*% (t(sv$u) / sv$d)
      X <- X %*% XP
      S <- t(XP) %*% S %*% XP
      # S <- S/eigen(S,symmetric=TRUE,only.values=TRUE)$values[1]
    }
  } else {
    XP <- NULL
  }

  return(list(X = X, knots = knots, S = S, D = D1, S.scale = maS, XP = XP))
}


WoodPenalty <- function(object1, object2) {
  # See Reiss et al. (2014) for details on the construction of the penalty matrices
  df1 <- ncol(object1$X)
  df2 <- ncol(object2$X)

  S1 <- object1$S %x% diag(df2)
  S2 <- diag(df1) %x% object1$S

  return(list(S1 = S1, S2 = S2))
}

WoodTensor <- function(X1, X2, coef.vector) {
  coef.matrix <- matrix(
    coef.vector,
    ncol = ncol(X1),
    nrow = ncol(X1),
    byrow = FALSE
  )

  spline <- X1 %*% coef.matrix %*% t(X2)

  return(spline)
}

WoodTensor.predict <- function(t1, t2, fit, logCRF = TRUE) {
  # Perform backtransform if repara = TRUE
  if (is.null(fit$splinepar[["XP1"]])) {
    X1 <- splines::splineDesign(
      fit$knots[[1]],
      t1,
      ord = fit$splinepar[["degree"]] + 1
    )
    X2 <- splines::splineDesign(
      fit$knots[[2]],
      t2,
      ord = fit$splinepar[["degree"]] + 1
    )
  } else {
    X1 <- splines::splineDesign(
      fit$knots[[1]],
      t1,
      ord = fit$splinepar[["degree"]] + 1
    ) %*%
      fit$splinepar[["XP1"]]
    X2 <- splines::splineDesign(
      fit$knots[[2]],
      t2,
      ord = fit$splinepar[["degree"]] + 1
    ) %*%
      fit$splinepar[["XP2"]]
  }

  # Model matrix
  X <- row_kron(X1, X2)
  spline <- X %*% fit$beta

  # Calculate standard error of log(theta)
  # var.logtheta <- X %*% fit$vcov %*% t(X)
  if (logCRF) {
    var.logtheta <- sapply(1:nrow(X), function(i) {
      t(X[i, ]) %*% fit$vcov %*% X[i, ]
    })
    se.logtheta <- sqrt(var.logtheta)
    return(data.frame(estimate = spline, se = se.logtheta))
  } else {
    return(exp(spline))
  }
}

# deriv_comp <- function(X1, X2, datalist) {
#
#   df <- ncol(X1)
#   M <- diag(df^2)
#
#   N1 <- datalist$riskset1
#   N2 <- datalist$riskset2
#
#   nrows <- length(N1)
#
#   deriv <- apply(M, 2,
#                  function(m) {A <- matrix(NA, ncol = 2, nrow = nrows)
#                               X <- WoodTensor(coef.vector = m, X1 = X1, X2 = X2)
#                               A[,1] <- c(t(X))[datalist$idxN1]
#                               A[,2] <- c(X)[datalist$idxN2]
#                               return(A) },
#                  simplify = FALSE
#                  )
#
#   return(deriv)
# }
#
# deriv_comp_poly <- function(datalist) {
#
#   df <- 10
#
#   M <- diag(df)
#
#   N1 <- datalist$riskset1
#   N2 <- datalist$riskset2
#
#   nrows <- sum(N1>0)
#
#   # List of gradient matrices for every spline coefficient
#   deriv <- apply(M, 2,
#                  function(m) {
#                    A <- matrix(NA, ncol = 2, nrow = nrows)
#                    B <- outer(datalist$X[,1], datalist$X[,2], function (x,y) polynomial(x,y, coef.vec = m))
#                    A[,1] <- c(t(B))[N1 > 0]
#                    A[,2] <- c(B)[N2 > 0]
#                    return(A)
#                  },
#                  simplify = FALSE)
#
#   return(deriv)
# }

# derivatives2 <- function(coef.vector, X1, X2, datalist, deriv, Sl = NULL, gradient = FALSE, hessian = TRUE) {
#
#   df <- ncol(X1)
#
#   # Tensor product spline
#   logtheta <- WoodTensor(X1, X2, coef.vector = coef.vector)
#   logtheta2 <- c(logtheta)[datalist$riskset2 > 0]
#   logtheta1 <- c(t(logtheta))[datalist$riskset1 > 0]
#
#   if (isTRUE(gradient)) {
#
#     gradient <- gradientC(riskset1 = datalist$riskset1[datalist$riskset1>0],
#                           riskset2 = datalist$riskset2[datalist$riskset2>0],
#                           logtheta1 = logtheta1,
#                           logtheta2 = logtheta2,
#                           deriv = deriv,
#                           df = df,
#                           delta1 = datalist$delta1,
#                           delta2 = datalist$delta2,
#                           I1 = datalist$I1,
#                           I2 = datalist$I2,
#                           I3 = datalist$I3,
#                           I4 = datalist$I4,
#                           I5 = datalist$I5,
#                           I6 = datalist$I6) # gradientC returns vector of derivatives of -loglik
#
#   } else {gradient <- NA}
#
#   if (isTRUE(hessian)) {
#
#     hessian <- hessianC(riskset1 = datalist$riskset1[datalist$riskset1>0],
#                         riskset2 = datalist$riskset2[datalist$riskset2>0],
#                         logtheta1 = logtheta1,
#                         logtheta2 = logtheta2,
#                         deriv = deriv,
#                         df = df,
#                         delta1 = datalist$delta1,
#                         delta2 = datalist$delta2,
#                         I1 = datalist$I1,
#                         I2 = datalist$I2,
#                         I3 = datalist$I3,
#                         I4 = datalist$I4) # hessianC returns matrix of second derivatives of -loglik
#
#   } else {hessian <- NA}
#
#   if (!is.null(Sl)) {
#     gradient <- gradient + t(coef.vector) %*% Sl
#     hessian <- hessian + Sl
#   }
#
#
#   return(list(gradient = gradient, hessian = hessian))
#   # return(gradient)
# }
#

Hessian <- function(
  coef.vector,
  X1,
  X2,
  Sl = NULL,
  datalist,
  ncores = 1,
  batch_size = 1000
) {
  # df <- ncol(X1)

  # Tensor product spline
  # logtheta <- WoodTensor(X1, X2, coef.vector = coef.vector)
  #
  # # print(logtheta[1:10, 1:10])
  #
  # logtheta2 <- c(logtheta)[datalist$idxN2+1]
  # logtheta1 <- c(t(logtheta))[datalist$idxN1+1]
  #
  # rm(logtheta)

  # hessianold <- hessianC(riskset1 = datalist$riskset1,
  #                     riskset2 = datalist$riskset2,
  #                     logtheta1 = logtheta1,
  #                     logtheta2 = logtheta2,
  #                     deriv = deriv,
  #                     df = df,
  #                     delta1 = datalist$delta1,
  #                     delta2 = datalist$delta2,
  #                     I1 = datalist$I1,
  #                     I2 = datalist$I2,
  #                     I3 = datalist$I3,
  #                     I4 = datalist$I4)

  if (ncores > 1) {
    hessian <- hessian_fast_batched_parallel(
      datalist = datalist,
      x = coef.vector,
      X1 = X1,
      X2 = X2,
      batch_size = batch_size
    )
  } else {
    hessian <- hessian_fast_batched(
      datalist = datalist,
      x = coef.vector,
      X1 = X1,
      X2 = X2,
      batch_size = batch_size
    )
  }

  if (!is.null(Sl)) {
    hessian <- hessian + Sl
  }

  return(hessian)
}


Score2 <- function(coef.vector, X1, X2, datalist, Sl = NULL, ncores = 1) {
  # Note that ncores = 1 has to be present for nleqsvl to work (same inputs needed as Hessian)

  # Tensor product spline
  # logtheta <- WoodTensor(X1, X2, coef.vector = coef.vector)
  # logtheta1 <- c(t(logtheta))[datalist$idxN1+1]
  # logtheta2 <- c(logtheta)[datalist$idxN2+1]
  #
  # rm(logtheta)

  gradient <- as.vector(gradient_fast(coef.vector, datalist, X1, X2)) # gradientC returns vector of derivatives of -loglik

  if (!is.null(Sl)) {
    penalty <- Sl %*% coef.vector
  } else {
    penalty <- 0
  }

  return(gradient + penalty)
}

# loglik.poly <- function(coef.vector) {
#
#   logtheta2 <- outer(X[,1], X[,2], function (x,y) polynomial(x,y, coef.vec = coef.vector))
#   logtheta1 <- t(logtheta2)
#
#   B1 <- c(I5*logtheta1)[N1 > 0]
#   B2 <- c(I6*logtheta2)[N2 > 0]
#
#   C1 <- c(N1 + I2*(exp(logtheta1)-1))[N1 > 0]
#   C2 <- c(N2 + I4*(exp(logtheta2)-1))[N2 > 0]
#
#
#   L1 <- sum(A1*(B1 - log(C1)))
#   L2 <- sum(A2*(B2 - log(C2)))
#
#   return(-(L1+L2))
# }

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

PrepareData <- function(t1, t2, cens1, cens2, ncores = 1) {
  RcppParallel::setThreadOptions(numThreads = ncores)

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

  delta.prod = delta(delta[, 1], delta[, 2])
  delta.prod1 <- c(t(delta.prod))
  delta.prod2 <- c(delta.prod)
  rm(delta.prod)

  RcppParallel::setThreadOptions(numThreads = 1)

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

wrapper2 <- function(
  coef.vector,
  X1,
  X2,
  datalist,
  Sl = NULL,
  H = NULL,
  minusLogLik = TRUE
) {
  # H is hier gewoon de unpenalized hessian

  # Check whether penalty is applied
  if (is.null(Sl)) {
    penaltyLik <- logSl <- logdetH <- 0
  } else {
    # Calculate penalty terms for log f_lambda(y,beta) Wood (2017) p.1076
    Sl.eigenv <- eigen(Sl, only.values = TRUE)$values
    Sl.eigenv[abs(Sl.eigenv) < 1e-6] <- 0

    penaltyLik <- t(coef.vector) %*% Sl %*% coef.vector
    logSl <- sum(log(Sl.eigenv[Sl.eigenv > 0]))
    if (is.infinite(logSl)) {
      stop("The log of the pseudo-determinant of Sl is ill-conditioned")
    }
  }

  if (!is.null(H)) {
    H.ev <- eigen(H + Sl, only.values = TRUE)$values
    H.ev.sign <- prod(sign(H.ev))
    logdetH <- H.ev.sign * sum(log(abs(H.ev)))
    if (is.infinite(logdetH)) {
      stop("The log of the determinant of H is ill-conditioned")
    }
  } else {
    logdetH <- 0
  }

  # logtheta <- WoodTensor(X1 = X1, X2 = X2, coef.vector = coef.vector)
  # logtheta1 <- c(t(logtheta))[datalist$idxN1+1]
  # logtheta2 <- c(logtheta)[datalist$idxN2+1]

  # rm(logtheta)

  L <- loglikC(coef.vector, datalist, X1, X2)

  ll <- L + penaltyLik / 2
  REML <- ll - logSl / 2 + logdetH / 2

  # Merk op dat C++ code geïmplementeerd is voor -loglik
  sign <- ifelse(isTRUE(minusLogLik), 1, -1)

  return(list(ll = sign * ll, REML = sign * REML))
}

# Create list of control parameters for EstimatePenal
efs.control <- function(
  lambda.tol = 0.1,
  REML.tol = 0.01,
  ll.tol = 0.01,
  maxiter = 100,
  lambda.max = exp(15),
  knot.margin = 0.001
) {
  list(
    lambda.tol = lambda.tol,
    REML.tol = REML.tol,
    ll.tol = ll.tol,
    maxiter = maxiter,
    lambda.max = lambda.max,
    knot.margin = knot.margin
  )
}

nleqslv.control <- function(method = "Broyden", global = "cline") {
  list(method = method, global = global)
}

# EFS gebaseerd op de code van Simon Wood in het mgcv package (zie gam.fit4.r op github)
# gam.control() details in mgcv.r op github
EstimatePenal2 <- function(
  datalist,
  dim,
  degree = 3,
  lambda.init = c(1, 1),
  start = rep(1, dim^2),
  # weights = NULL,
  type = "ps",
  quantile = FALSE,
  scale = TRUE,
  repara = FALSE,
  observed.region = FALSE,
  step.control = FALSE,
  control = efs.control(),
  nl.control = nleqslv.control(),
  verbose = TRUE,
  ncores = 1
) {
  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1))

  if (verbose) {
    print("Extended Fellner-Schall method:")
  }

  # if (is.null(weights)) weights <- rep(1, length(datalist$riskset1))

  tiny <- .Machine$double.eps^0.5

  obj1 <- WoodSpline(
    t = datalist$X[, 1],
    delta = datalist$delta[, 1],
    dim = dim,
    degree = degree,
    type = type,
    scale = scale,
    repara = repara,
    observed.region = observed.region,
    quantile = quantile,
    knot.margin = control$knot.margin
  )
  obj2 <- WoodSpline(
    t = datalist$X[, 2],
    delta = datalist$delta[, 2],
    dim = dim,
    degree = degree,
    type = type,
    scale = scale,
    repara = repara,
    observed.region = observed.region,
    quantile = quantile,
    knot.margin = control$knot.margin
  )

  S <- WoodPenalty(obj1, obj2)
  S1 <- S[[1]]
  S2 <- S[[2]]

  lambda.new <- lambda.init # In voorbeelden van Wood (2017) is de initiele lambda = 1

  # deriv.comp <- deriv_comp(X1 = X1, X2 = X2, datalist = datalist)

  fit <- efsud.fit2(
    start = start,
    X1 = obj1$X,
    X2 = obj2$X,
    datalist = datalist,
    # Sl = lambda.init*S
    Sl = lambda.init[1] * S1 + lambda.init[2] * S2,
    control = nl.control,
    ncores = ncores
  )
  k <- 1
  score <- rep(0, control$maxiter)
  for (iter in 1:control$maxiter) {
    l0 <- fit$REML

    lambda <- lambda.new

    # Some calculations to update lambda later...
    Sl <- lambda[1] * S1 + lambda[2] * S2
    # Sl <- lambda*S
    Sl.inv <- MASS::ginv(Sl)

    # decomp <- eigen(hessian)
    # A <- diag(abs(decomp$values))
    # hessian <- decomp$vectors %*% A %*% t(decomp$vectors)

    # Update ----

    # Calculate V
    V <- solve(fit$hessian + Sl)

    # Calculate trSSj, trVS and bSb
    trSSj <- trVS <- bSb <- rep(0, length(S))
    for (i in 1:length(S)) {
      trSSj[i] <- sum(diag(Sl.inv %*% S[[i]]))
      trVS[i] <- sum(diag(V %*% S[[i]]))
      bSb[i] <- t(fit$beta) %*% S[[i]] %*% fit$beta
    }

    # trSSj <- sum(diag(Sl.inv %*% S))
    # trVS <- sum(diag(V %*% S))
    # bSb <- t(fit$beta) %*% S %*% fit$beta

    # Update lambdas
    a <- pmax(tiny, trSSj - trVS)
    update <- a / pmax(tiny, bSb)
    update[a == 0 & bSb == 0] <- 1
    update[!is.finite(update)] <- 1e6
    lambda.new <- pmin(update * lambda, control$lambda.max)

    # Step length of update
    max.step <- max(abs(lambda.new - lambda))

    # Create new S.lambda matrix
    Sl.new <- lambda.new[1] * S1 + lambda.new[2] * S2
    # Sl.new <- lambda.new*S

    fit <- efsud.fit2(
      start = fit$beta,
      X1 = obj1$X,
      X2 = obj2$X,
      datalist = datalist,
      Sl = Sl.new,
      control = nl.control,
      ncores = ncores
    )
    l1 <- fit$REML

    # Start of step control ----
    if (step.control) {
      if (l1 > l0) {
        # Improvement
        if (max.step < 1) {
          # Consider step extension
          lambda2 <- pmin(lambda * update^(k * 2), exp(12))
          fit2 <- efsud.fit2(
            start = fit$beta,
            X1 = obj1$X,
            X2 = obj2$X,
            datalist = datalist,
            # Sl = lambda2*S
            Sl = lambda2[1] * S1 + lambda2[2] * S2,
            # weights = weights,
            control = nl.control,
            ncores = ncores
          )
          l2 <- fit2$REML
          if (l2 > l1) {
            # Improvement - accept extension
            lambda.new <- lambda2
            l1 <- l2
            fit <- fit2
            k <- k * 2
          }
        }
      } else {
        # No improvement
        lk <- l1
        lambda3 <- lambda.new
        while (lk < l0 && k > 1) {
          # Don't contract too much since the likelihood does not need to increase k > 0.001
          k <- k / 2 ## Contract step
          lambda3 <- pmin(lambda * update^k, control$lambda.max)
          fit <- efsud.fit2(
            start = fit$beta,
            X1 = obj1$X,
            X2 = obj2$X,
            datalist = datalist,
            # Sl = lambda3*S
            Sl = lambda3[1] * S1 + lambda3[2] * S2,
            # weights = weights,
            control = nl.control,
            ncores = ncores
          )
          lk <- fit$REML

          # k <- k + 1
          # diff <- diff/2
          # lambda3 <- lambda + diff
          # lk <- wrapper(coef.vector = beta, degree = degree,
          #               # Sl = lambda3[1]*S1 + lambda3[2]*S2,
          #               Sl = lambda3*S,
          #               H = hessian, minusLogLik = FALSE, datalist = datalist)
        }
        lambda.new <- lambda3
        l1 <- lk
        max.step <- max(abs(lambda.new - lambda))
        if (k < 1) k <- 1
      }
    } # end of step length control

    # save loglikelihood value
    score[iter] <- l1

    # Print information while running...
    if (verbose) {
      print(paste0(
        "Iteration ",
        iter,
        ": k = ",
        k,
        # " lambda = ", round(lambda.new,4),
        " lambda1 = ",
        round(lambda.new[1], 4),
        " lambda2 = ",
        round(lambda.new[2], 4),
        " ll = ",
        round(fit$ll, 4),
        " REML = ",
        score[iter]
      ))
    }

    # Break procedures ----

    # Break procedure if REML change and step size are too small
    if (
      iter > 3 &&
        max(abs(diff(score[(iter - 3):iter]))) < control$REML.tol &&
        max.step < control$lambda.tol
    ) {
      if (verbose) {
        print("REML not changing")
      }
      break
    }
    # Or break is likelihood does not change
    # if (l1 == l0) {if (verbose) print("Loglik not changing"); break}

    # Stop if loglik is not changing
    if (iter == 1) {
      old.ll <- fit$ll
    } else {
      if (abs(old.ll - fit$ll) < control$ll.tol) {
        if (verbose) {
          print("Loglik not changing")
        }
        break
      } # if (abs(old.ll-fit$ll)<100*eps*abs(fit$ll))
      old.ll <- fit$ll
    }
  } # End of for loop

  if (verbose) {
    if (iter < control$maxiter) {
      print("Converged")
    } else {
      print("Maximum number of iterations reached")
    }
  }

  return(list(
    beta = fit$beta,
    lambda = lambda.new,
    vcov = 2 * solve(fit$hessian + lambda.new[1] * S1 + lambda.new[2] * S2),
    iterations = iter,
    ll = fit$ll,
    history = score[1:iter],
    info = fit$estim,
    splinepar = list(dim = dim, degree = degree, XP1 = obj1$XP, XP2 = obj2$XP),
    knots = list(knots1 = obj1$knots, knots2 = obj2$knots)
  ))
}


efsud.fit2 <- function(
  start,
  X1,
  X2,
  datalist,
  Sl,
  control = nleqslv.control(),
  ncores = 1
) {
  # if (is.null(deriv.comp)) deriv <- deriv_comp(X1 = X1, X2 = X2, datalist = datalist, weights = weights)
  # else deriv <- deriv.comp

  # beta <- multiroot(Score2, start = start, jacfunc = Hessian, jactype = "fullusr", rtol = 1e-10, X1 = X1, X2 = X2, Sl = Sl, datalist = datalist, deriv = deriv)$root

  estim <- nleqslv::nleqslv(
    x = start,
    fn = Score2,
    jac = Hessian,
    method = control$method,
    global = control$global,
    X1 = X1,
    X2 = X2,
    datalist = datalist,
    Sl = Sl,
    ncores = ncores
  )
  beta <- estim$x
  if (any(is.na(beta))) {
    estim
    stop("One of the spline coefficients is NA")
  }

  H <- Hessian(
    coef.vector = beta,
    X1 = X1,
    X2 = X2,
    datalist = datalist,
    ncores = ncores
  )

  fit <- wrapper2(
    coef.vector = beta,
    X1 = X1,
    X2 = X2,
    Sl = Sl,
    H = H,
    minusLogLik = FALSE,
    datalist = datalist
  )

  return(list(
    beta = beta,
    hessian = H,
    REML = fit$REML,
    ll = fit$ll,
    info = estim
  ))
}
