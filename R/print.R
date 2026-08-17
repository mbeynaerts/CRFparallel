# #' @export
# print.CRF <- function(object, digits = getOption(digits), ...) {
#   glue::glue(
#     "
#     Call:
#     {deparse(object$call)}"
#   )
#   if (inherits(object, "CRFbern")) {
#     glue::glue("Bernstein order: m = {object$m}")
#   } else {
#     glue::glue(
#       "
#       Coefficients:
#       {format(round(object, digits), nsmall = digits)}"
#     )
#   }
#   invisible(object)
# }

print <- S7::new_external_generic("base", "print", "x")

#' @export
S7::method(print, CRFspline) <- function(x, digits = getOption(digits)) {
  glue::glue(
    "
    Call: 
    {deparse(x@call)}"
  )
  glue::glue(
    "
    Coefficients:
    {format(round(x@coefficients@coefficients, digits), nsmall = digits)}"
  )
  invisible(x)
}

#' @export
S7::method(print, CRFpoly) <- function(x, digits = getOption(digits)) {
  glue::glue(
    "
    Call: 
    {deparse(x@call)}"
  )
  glue::glue(
    "
    Coefficients:
    {format(round(x@coefficients@coefficients, digits), nsmall = digits)}"
  )
  invisible(x)
}

#' @export
S7::method(print, CRFbern) <- function(x) {
  glue::glue(
    "
    Call: 
    {deparse(x@call)}"
  )
  glue::glue("Bernstein order: m = {x@m}")
  invisible(x)
}
