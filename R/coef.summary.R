#' @method coef summary
#' @export
coef.summary <- function(object, ...) {
  object$coefficients
}
