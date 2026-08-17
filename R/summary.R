summary <- S7::new_external_generic("base", "summary", "object")

S7::method(summary, CRFspline) <- function(object) {
  se <- sqrt(diag(object@vcov))
  summary_list <- list(
    call = object@call,
    method = "spline",
    coefficients = coef(object),
    se = se,
    p025 = coef(object) - 1.96 * se,
    p975 = coef(object) + 1.96 * se,
    degree = attr(object@splines2[[1]], "degree"),
    dim = ncol(object@model.matrix@model.matrix),
  )
  class(summary_list) <- "summary"
  return(summary_list)
}

S7::method(summary, CRFpoly) <- function(object) {
  se <- sqrt(diag(object@vcov))
  summary_list <- list(
    call = object@call,
    method = "polynomial",
    degree = object@degree,
    restrict_degree = object@restrict_degree,
    coefficients = coef(object),
    se = se,
    p025 = coef(object) - 1.96 * se,
    p975 = coef(object) + 1.96 * se
  )
  class(summary_list) <- "summary"
  return(summary_list)
}

# summary.CRFspline <- function(object, ...) {
#   if (!inherits(object, 'CRFspline')) {
#     cli::cli_abort(
#       "{.fun summary.CRFspline} can only be used for {.cls CRFspline} objects"
#     )
#   }
#   se <- sqrt(diag(object$vcov))
#   summary_list <- list(
#     call = object$call,
#     method = "spline",
#     type = object$type,
#     coefficients = coef(object),
#     se = se,
#     p025 = coef(object) - 1.96 * se,
#     p975 = coef(object) + 1.96 * se,
#     type = object$type,
#     degree = object$degree,
#     dim = object$dim,
#   )
#   class(summary_list) <- "summary.CRFspline"
#   return(summary_list)
# }

# summary.CRFpoly <- function(object, ...) {
#   if (!inherits(object, 'CRFpoly')) {
#     cli::cli_abort(
#       "{.fun summary.CRFpoly} can only be used for {.cls CRFpoly} objects"
#     )
#   }
#   se <- sqrt(diag(object$vcov))
#   summary_list <- list(
#     call = object$call,
#     method = "polynomial",
#     degree = object$degree,
#     restrict_degree = object$restrict_degree,
#     coefficients = coef(object),
#     se = se,
#     p025 = coef(object) - 1.96 * se,
#     p975 = coef(object) + 1.96 * se
#   )
#   class(summary_list) <- "summary.CRFpoly"
#   return(summary_list)
# }
