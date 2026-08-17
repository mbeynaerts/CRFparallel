reml_spline <- function(
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
      cli::cli_abort(
        "The log of the pseudo-determinant of Sl is ill-conditioned"
      )
    }
  }

  if (!is.null(H)) {
    H.ev <- eigen(H + Sl, only.values = TRUE)$values
    H.ev.sign <- prod(sign(H.ev))
    logdetH <- H.ev.sign * sum(log(abs(H.ev)))
    if (is.infinite(logdetH)) {
      cli::cli_abort("The log of the determinant of H is ill-conditioned")
    }
  } else {
    logdetH <- 0
  }

  L <- loglikC(coef.vector, datalist, X1, X2)

  ll <- L + penaltyLik / 2
  REML <- ll - logSl / 2 + logdetH / 2

  # Merk op dat C++ code geïmplementeerd is voor -loglik
  sign <- ifelse(isTRUE(minusLogLik), 1, -1)

  return(list(ll = sign * ll, REML = sign * REML))
}
