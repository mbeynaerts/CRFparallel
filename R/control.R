#' Initialize control parameters of the Generelized Fellner-Schall method
#'
#' @param lambda.tol Absolute tolerance for the difference in smoothing parameters between subsequent iterations. Default is `0.1`.
#' @param REML.tol Absolute tolerance for the difference in Restricted Maximum Likelihood between subsequent iterations.
#' @param ll.tol Absolute tolerance for the difference in Restricted Maximum Likelihood between subsequent iterations. Default is `0.01`.
#' @param maxiter Maximum number of iterations. Default is `100`.
#' @param lambda.max Maximum value of the smoothing parameters. A smoothing parameter exceedings this value will terminate the Fellner-Schall algorithm. Default is `exp(15)`.
#' @param knot.margin description
#' @param step.control description
#'
#' @returns A named list of control parameters for the generalized Fellner-Schall method.
#'
#' @export
efs.control <- function(
  lambda.tol = 0.1,
  REML.tol = 0.01,
  ll.tol = 0.01,
  maxiter = 100L,
  lambda.max = exp(15),
  knot.margin = 0.001,
  step.control = FALSE
) {
  list(
    lambda.tol = lambda.tol,
    REML.tol = REML.tol,
    ll.tol = ll.tol,
    maxiter = maxiter,
    lambda.max = lambda.max,
    knot.margin = knot.margin,
    step.control = step.control
  )
}

#' Initialize control parameters for `nlesqslv` solvers
#'
#' @param method A character string specifying the nonlinear equation solver to use. See the `method` argument in [nleqslv::nleqslv()] for details. Default is "Broyden".
#' @param global A character string specifying the global strategy. See the `global` argument in [nleqslv::nleqslv()] for details. Default is "cline".
#' @param ... Any other arguments passed to [nleqslv::nleqslv()]. See the `control` argument in [nleqslv::nleqslv()] for details.
#'
#' @returns A named list of control parameters.
#'
#' @export
nleqslv.control <- function(method = "Broyden", global = "cline", ...) {
  list(method = method, global = global, ...)
}

#' Initializing spline parameters
#'
#' @param type A character string specifying the type of penalty applied to the B-spline basis. Options are `"ps"` for B-splines with second-order difference penalty (P-splines), `"bs"` for B-splines with integrated squared derivative penalty, and `"gps"` for generalized P-splines. Default is `"ps"`.
#' @param degree An integer specifying the degree of the spline basis. Default is 3.
#' @param dim An integer specifying the number of basis functions to use in each dimension. Default is 10.
#' @param start An optional numeric vector specifying the starting values of the spline coefficients. Default is a vector of 1s with length equal to `dim^2`.
#' @param lambda.init Numeric vector with the two initial smoothing parameters. Default is `c(1, 1)`.
#' @param scale Logical indicating whether to scale the penalty matrices for numerical stability. Default is `TRUE`.
#' @param quantile Logical indicating whether to use quantile-based knot placement. Default is `FALSE`. Note that quantile-based knots are not supported for spline of type `"ps"`.
#' @param observed.region Logical indicating whether knot placement should be based on the observed data only (`TRUE`) or on the entire data including censored observations (`FALSE`). Currently `TRUE`is not supported and defaults to `FALSE`.
#' @param knots An optional list of knot locations for each dimension. If `NULL`, knots will be placed automatically based on the specified number of basis functions and selected penalty. Default is `NULL`.
#' @param knot.margin Relative margin added to the observed time range for boundary knots. Default is `0.001`.
#'
#' @returns A named list of control parameters for spline-based estimation.
#'
#' @export
spline.control <- function(
  type = c("ps", "bs", "gps"),
  degree = 3L,
  dim = 10L,
  start = rep(1, dim^2),
  lambda.init = c(1, 1),
  scale = TRUE,
  quantile = FALSE,
  observed.region = FALSE, # TRUE not supported
  knots = NULL,
  knot.margin = 0.001
) {
  type <- match.arg(type)

  if (degree < 1) {
    cli::cli_abort("{.arg degree} must be a positive integer.")
  }

  if (dim < 1) {
    cli::cli_abort("{.arg dim} must be a positive integer.")
  }

  if (isFALSE(all.equal(degree %% 1, 0))) {
    cli::cli_abort("{.arg degree} must be an integer.")
  } else {
    degree <- as.integer(degree)
  }

  if (isFALSE(all.equal(dim %% 1, 0))) {
    cli::cli_abort("{.arg degree} must be an integer.")
  } else {
    dim <- as.integer(dim)
  }

  if (!is.numeric(knot.margin) || length(knot.margin) != 1 || knot.margin < 0) {
    cli::cli_abort("{.arg knot.margin} must be a non-negative number.")
  }

  if (!is.numeric(lambda.init) || length(lambda.init) != 2) {
    cli::cli_abort("{.arg lambda.init} must be a numeric vector of length 2.")
  }

  #TODO create check for dim > degree

  # Check equidistan knot placement for "ps"
  if (type == "ps" & quantile) {
    quantile <- FALSE
    cli::cli_warn(
      "Quantile-based knots are not supported for type `ps`. {.arg quantile} set to `FALSE`"
    )
  }

  if (!is.null(knots)) {
    if (!is.list(knots)) {
      cli::cli_abort(
        "Knots must be provided as a list where each element correponds to a vector of interior knots."
      )
    } else {
      check_knots(knots, type, dim, degree)
    }
  }

  list(
    type = type,
    degree = degree,
    dim = dim,
    start = start,
    lambda.init = lambda.init,
    scale = scale,
    quantile = quantile,
    observed.region = FALSE,
    knots = knots,
    knot.margin = knot.margin
  )
}


#' Initializing polynomial parameters
#'
#' @param degree description
#' @param restrict_degree description
#'
#' @returns A named list of control parameters for polynomial maximum likelihood estimation.
#'
#' @export
polynomial.control <- function(
  degree = 3L,
  restrict_degree = 3L
) {
  if (degree < 1) {
    cli::cli_abort("{.arg degree} must be a positive integer.")
  }

  if (restrict_degree < 1) {
    cli::cli_abort("{.arg restrict_degree} must be a positive integer.")
  }

  if (isFALSE(all.equal(degree %% 1, 0))) {
    cli::cli_abort("{.arg degree} must be an integer.")
  } else {
    degree <- as.integer(degree)
  }

  if (isFALSE(all.equal(restrict_degree %% 1, 0))) {
    cli::cli_abort("{.arg restrict_degree} must be an integer.")
  } else {
    restrict_degree <- as.integer(restrict_degree)
  }

  list(
    degree = degree,
    restrict_degree = restrict_degree
  )
}


#' Title
#'
#' @param m An integer specifying the Bernstein order. Default is 10.
#'
#' @returns A named list of control parameters for the Bernstein method.
#'
#' @export
bernstein.control <- function(m = 10L) {
  if (m < 1) {
    cli::cli_abort("Bernstein order {.arg m} must be a positive integer.")
  }
  if (m %% 1 != 0) {
    # Check if m can be coerced to an integer without rounding
    cli::cli_abort("Bernstein order {.arg m} must be an integer.")
  } else {
    m <- as.integer(m) # Ensure m is an integer
  }

  list(m = m) # Currently only fixed order until order selection is implemented
}
