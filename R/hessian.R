hessian_spline <- function(
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
  # logtheta <- tensor_product(X1, X2, coef.vector = coef.vector)
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


hessian_poly <- function(beta, X1, X2, datalist, idx) {
  hessian_poly_batched_parallel(beta, datalist, idx, X1, X2)
}
