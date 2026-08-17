# #' @export
# coef.CRF <- function(object, ...) {
#   if (inherits(object, "CRFbern")) {
#     cli::cli_abort(
#       "The {.arg coef} method is not currently implemented for objects of class {.cls CRFbern}."
#     )
#   } else {
#     x <- object$coefficients
#     names(x) <- glue::glue("beta{1:length(object$coefficients)}")
#   }

#   return(x)
# }

coef <- S7::new_external_generic("stats", "coef", "object")

#' @export
S7::method(coef, CRFspline) <- function(object) {
  y <- object@coefficients@coefficients
  names(y) <- object@coefficients@label
  return(y)
}

#' @export
S7::method(coef, CRFpoly) <- function(object) {
  y <- object@coefficients@coefficients
  names(y) <- object@coefficients@label
  return(y)
}
