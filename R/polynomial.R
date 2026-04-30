keep_indices <- function(poly_degree = 3, restrict_degree = 3) {
  dim_mat <- poly_degree + 1

  # Create a dummy matrix to find the indices
  # We use column-major order to match Armadillo/R defaults
  dummy_mat <- matrix(0, nrow = dim_mat, ncol = dim_mat)
  keep_indices <- c()

  k <- 0
  for (j in 1:dim_mat) {
    for (i in 1:dim_mat) {
      # Adjust for R's 1-based indexing: (i-1) + (j-1) <= poly_order
      if ((i - 1) + (j - 1) <= restrict_degree) {
        # This is the linear index (0-based for C++)
        keep_indices <- c(keep_indices, k)
      }
      k <- k + 1
    }
  }

  # Output: indices of coefficients to keep in the (poly_degree+1) x (poly_degree+1) coefficient matrix
  # These indices are C++ indices, for R use keep_indices+1
  return(keep_indices)
}

gradient.poly <- function(beta, X1, X2, datalist, idx) {
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

hessian.poly <- function(beta, X1, X2, datalist, idx) {
  hessian_poly_batched_parallel(beta, datalist, idx, X1, X2)
}
