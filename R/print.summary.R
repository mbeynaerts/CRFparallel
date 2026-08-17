#' @method print summary
#' @export
print.summary <- function(x, digits = getOption("digits"), ...) {
  summary(x, ...)
}
