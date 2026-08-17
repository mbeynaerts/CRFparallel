fitted.values <- S7::new_external_generic("stats", "fitted.values", "object")

fitted <- S7::new_external_generic("stats", "fitted", "object")

#' @export
S7::method(fitted.values, CRFspline) <- function(object) object@fitted.values

#' @export
S7::method(fitted.values, CRFpoly) <- function(object) object@fitted.values

#' @export
S7::method(fitted.values, CRFbern) <- function(object) object@fitted.values

#' @export
S7::method(fitted, CRFspline) <- function(object) object@fitted.values

#' @export
S7::method(fitted, CRFpoly) <- function(object) object@fitted.values

#' @export
S7::method(fitted, CRFbern) <- function(object) object@fitted.values
