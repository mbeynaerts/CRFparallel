gradient_spline <- function(
  coef.vector,
  X1,
  X2,
  datalist,
  Sl = NULL,
  ncores = 1
) {
  # Note that ncores = 1 has to be present for nleqsvl to work (same inputs needed as hessian_spline)

  # Tensor product spline
  # logtheta <- tensor_product(X1, X2, coef.vector = coef.vector)
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

gradient_poly <- function(beta, X1, X2, datalist, idx) {
  gradient <- gradient_poly_fast(beta, datalist, idx, X1, X2) # gradientC returns vector of derivatives of -loglik

  return(as.vector(gradient))

  # logtheta1 <- t(logtheta2)
  #
  # N1 <- t(datalist$riskset)
  # N2 <- datalist$riskset
  #
  # delta2 <- datalist$delta.prod
  # delta1 <- t(delta2)
  #
  # I1 <- datalist$I1
  # I2 <- datalist$I2
  # I3 <- t(datalist$I2)
  # I4 <- t(datalist$I1)
  # I5 <- datalist$I5
  # I6 <- datalist$I6
  #
  # A1 <- (delta1*I1)[N1 > 0]
  # A2 <- (delta2*I3)[N2 > 0]
  #
  # B1 <- c(datalist$I5*logtheta1)[N1 > 0]
  # B2 <- c(datalist$I6*logtheta2)[N2 > 0]
  #
  # C1 <- c(N1 + I2*(exp(logtheta1)-1))[N1 > 0]
  # C2 <- c(N2 + I4*(exp(logtheta2)-1))[N2 > 0]
  #
  #
  # L1 <- sum(A1*(B1 - log(C1)))
  # L2 <- sum(A2*(B2 - log(C2)))
  #
  # L1 <- logLikC(riskset = t(datalist$riskset),
  #               logtheta = t(logtheta2),
  #               delta = t(datalist$delta.prod),
  #               I1 = datalist$I1, I2 = datalist$I2, I3 = datalist$I5)
  #
  # L2 <- logLikC(riskset = datalist$riskset,
  #               logtheta = logtheta2,
  #               delta = datalist$delta.prod,
  #               I1 = t(datalist$I2), I2 = t(datalist$I1), I3 = datalist$I6)
  #
  #   return(-L1-L2)
}
