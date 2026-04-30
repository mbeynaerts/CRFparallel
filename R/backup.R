Poly.predict <- function(t1, t2, fit, logCRF = TRUE) {
  X1 <- cbind(1, t1, t1^2, t1^3)
  X2 <- cbind(1, t2, t2^2, t2^3)

  ord <- fit$degree + 1

  beta <- rep(0, ord^2)
  beta[fit$idx] <- fit$beta

  logtheta <- row_kron(X1, X2) %*% beta

  if (logCRF) {
    return(logtheta)
  } else {
    return(exp(logtheta))
  }
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
    # var.logtheta <- sapply(1:nrow(X), function(i) {
    #   t(X[i, ]) %*% fit$vcov %*% X[i, ]
    # })
    var.logtheta <- rowSums((X %*% fit$vcov) * X)
    se.logtheta <- sqrt(var.logtheta)
    return(data.frame(estimate = spline, se = se.logtheta))
  } else {
    return(exp(spline))
  }
}

polynomial <- function(t1, t2, coef.vec, logCRF = TRUE) {
  logtheta <- coef.vec[1] +
    coef.vec[2] * t1 +
    coef.vec[3] * t2 +
    coef.vec[4] * t1^2 +
    coef.vec[5] * t2^2 +
    coef.vec[6] * t1 * t2 +
    coef.vec[7] * (t1^2) * t2 +
    coef.vec[8] * t1 * (t2^2) +
    coef.vec[9] * t1^3 +
    coef.vec[10] * t2^3

  if (logCRF) {
    return(logtheta)
  } else {
    return(exp(logtheta))
  }
}
