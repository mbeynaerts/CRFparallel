CRF.object <- function(method = c("spline", "polynomial"), method.args) {
  stopifnot(is.list(method.args))
  method <- match.arg(method)
  object <- list()

  if (method == "spline" || method == "polynomial") {
    object$model.matrix <- matrix()
    object$fitted.values <- numeric()
    object$vcov <- matrix()
    object$method <- method

    if (method == "spline") {
      class(object) <- "CRFspline"
      object$coefficients <- numeric(length = method.args$dim^2)
      object$knots <- list
      attr(object$coefficients, "names") <- paste0("beta", 1:method.args$dim^2)
      attr(object$model.matrix, "idx") <- 1:(method.args$dim^2)
      attr(object$method, "type") <- method.args$type
      attr(object$method, "dim") <- method.args$dim
      attr(object$method, "degree") <- method.args$degree
      attr(object$method, "scale") <- method.args$scale
      attr(object$method, "lambda") <- numeric(length = 2)
      attr(object$method, "knots") <- vector("list", length = 2)
      attr(object$method, "quantile") <- method.args$quantile
      attr(object$method, "iterations") <- integer()
    } else if (method == "polynomial") {
      class(object) <- "CRFpoly"
      l <- (method.args$restrict_degree + 1) *
        (method.args$restrict_degree + 2) /
        2
      object$coefficients <- numeric(length = l)
      attr(object$coefficients, "names") <- paste0("beta", 1:l)
      attr(object$model.matrix, "idx") <- integer()
      attr(object$method, "degree") <- method.args$degree
      attr(object$method, "restrict_degree") <- method.args$restrict_degree
    }
    object$loglik <- numeric()
    object$reml <- ifelse(method == "spline", numeric(), NA)
    object$model <- data.frame()
  } else if (method == "bernstein") {
    # TODO - Implement CRF object "CRFbernstein"
    class(object) <- "CRFbernstein"
    object$fitted.values <- numeric()
    object$m <- method.args$m
    object$tau <- numeric(length = 2)
  } else {
    stop(
      "Method not supported. Please choose 'spline', 'polynomial' or 'bernstein'."
    )
  }

  return(object)
}
