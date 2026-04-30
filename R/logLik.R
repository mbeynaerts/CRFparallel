logLik.CRFspline <- function(object, REML = FALSE, ...) {
  if (!inherits(object, "CRFspline")) {
    stop("Object must be of class 'CRFspline'.")
  }

  # Extract log-likelihood from the CRFspline object
  if (!REML) {
    loglik <- object$loglik
  } else {
    loglik <- object$reml
  }

  # Return log-likelihood as an object of class 'logLik'
  structure(loglik, class = "logLik", df = length(coef(object)))
}

logLik.CRFpoly <- function(object, ...) {
  if (!inherits(object, "CRFpoly")) {
    stop("Object must be of class 'CRFpoly'.")
  }

  # Extract log-likelihood from the CRFpoly object
  loglik <- object$loglik

  # Return log-likelihood as an object of class 'logLik'
  structure(loglik, class = "logLik", df = length(coef(object)))
}
