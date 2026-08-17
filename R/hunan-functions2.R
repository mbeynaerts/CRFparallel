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
#                               X <- tensor_product(coef.vector = m, X1 = X1, X2 = X2)
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
#   logtheta <- tensor_product(X1, X2, coef.vector = coef.vector)
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
