logLik <- S7::new_external_generic("stats", "logLik", "object")

#' @export
S7::method(logLik, CRFspline) <- function(object, REML = FALSE) {
  # Extract log-likelihood from the CRFspline object
  if (!REML) {
    loglik <- object@loglik
  } else {
    loglik <- object@reml
  }

  # Return log-likelihood as an object of class 'logLik'
  structure(
    loglik,
    class = "logLik",
    df = length(object@coefficients@coefficients)
  )
}

#' @export
S7::method(logLik, CRFpoly) <- function(object) {
  # Extract log-likelihood from the CRFpoly object
  loglik <- object@loglik

  # Return log-likelihood as an object of class 'logLik'
  structure(
    loglik,
    class = "logLik",
    df = length(object@coefficients@coefficients)
  )
}
