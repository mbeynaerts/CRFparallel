

poly.fit <- function (beta, X1, X2, datalist) {
  
  gradient <- gradient_poly_fast(beta,
                                 datalist,
                                 X1,
                                 X2) # gradientC returns vector of derivatives of -loglik
  
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

HessianPoly <- function(beta, X1, X2, datalist) {
  
  hessian <- hessian_poly_batched_parallel(beta, X1, X2, datalist)
  return(hessian)
}


EstimatePoly <- function(start = rep(0,10), datalist, control = nleqslv.control(), ncores = 1) {
  RcppParallel::setThreadOptions(numThreads = ncores)
  
  
  X1 <- cbind(1,
              datalist$X[,1],
              datalist$X[,1],
              datalist$X[,1]^2,
              datalist$X[,1]^3)
  X2 <- cbind(1,
              datalist$X[,2],
              datalist$X[,2],
              datalist$X[,2]^2,
              datalist$X[,2]^3)
  
  beta <- nleqslv::nleqslv(x = start, fn = poly.fit, jac = HessianPoly,
                           method = control$method, global = control$global, datalist = datalist, X1 = X1, X2 = X2)
  
  V <- HessianPoly(beta$x, datalist, deriv)
  
  RcppParallel::setThreadOptions(numThreads = ncores)
  
  return(list(beta = beta$x, vcov = solve(V)))
  
}