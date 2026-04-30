estimate_poly <- function(
  datalist,
  start = rep(0, 10),
  degree = 3,
  restrict_degree = 3,
  poly.control = polynomial.control(),
  nl.control = nleqslv.control(),
  ncores = 1
) {
  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1), add = TRUE)

  # X1 <- cbind(1, datalist$X[, 1], datalist$X[, 1]^2, datalist$X[, 1]^3)
  # X2 <- cbind(1, datalist$X[, 2], datalist$X[, 2]^2, datalist$X[, 2]^3)

  final <- CRF.object(method = "polynomial", method.args = poly.control)
  stopifnot(inherits(final, "CRFpoly")) # Check that final is of class CRFpoly

  X1 <- model.matrix(
    ~ poly(datalist$X[, 1], degree = d, raw = TRUE, simple = TRUE)
  )
  X2 <- model.matrix(
    ~ poly(datalist$X[, 2], degree = d, raw = TRUE, simple = TRUE)
  )

  idx <- keep_indices(
    poly_degree = poly_degree,
    restrict_degree = restrict_degree
  )
  final$model.matrix <- row_kron(X1, X2)
  attr(final$model.matrix, "idx") <- idx + 1

  beta <- nleqslv::nleqslv(
    x = start,
    fn = gradient.poly,
    jac = hessian.poly,
    method = nl.control$method,
    global = nl.control$global,
    idx = idx,
    datalist = datalist,
    X1 = X1,
    X2 = X2
  )

  V <- hessian.poly(beta$x, X1 = X1, X2 = X2, datalist = datalist, idx = idx)

  coef.matrix <- matrix(0.0, nrow = degree + 1, ncol = degree + 1)
  coef.matrix[idx + 1] <- beta$x

  final$fitted.values <- X1 %*% coef.matrix %*% t(X2)
  final$coefficients <- beta$x
  final$vcov <- solve(V)
  final$loglik <- beta$fvec

  # return(list(beta = beta$x, vcov = solve(V), degree = degree, idx = idx + 1))
  return(final)
}
