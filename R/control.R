efs.control <- function(
  lambda.tol = 0.1,
  REML.tol = 0.01,
  ll.tol = 0.01,
  maxiter = 100,
  lambda.max = exp(15),
  knot.margin = 0.001
) {
  list(
    lambda.tol = lambda.tol,
    REML.tol = REML.tol,
    ll.tol = ll.tol,
    maxiter = maxiter,
    lambda.max = lambda.max,
    knot.margin = knot.margin
  )
}

nleqslv.control <- function(method = "Broyden", global = "cline") {
  list(method = method, global = global)
}

spline.control <- function(
  type = c("ps", "bs", "gps"),
  degree = 3,
  dim = 10,
  start = rep(1, dim^2),
  scale = TRUE,
  quantile = FALSE,
  observed.region = FALSE,
  step.control = FALSE,
  knots = NULL
) {
  type <- match.arg(type)
  if (type == "ps" & quantile) {
    # Check equidistan knot placement for "ps"
    warning(
      "'quantile' set to FALSE for 'ps' as quantile-based knots are not supported for P-splines."
    )
    quantile <- FALSE
  }

  if (!is.list(knots) & !is.null(knots)) {
    stop("Knots must be provided as a list.")
  }
  #TODO - sanity check for knots with dim, degree and data range

  list(
    type = type,
    degree = degree,
    dim = dim,
    start = start,
    scale = scale,
    quantile = quantile,
    observed.region = FALSE,
    step.control = step.control,
    knots = knots
  )
}

polynomial.control <- function(
  degree = 3,
  restrict_degree = 3
) {
  list(
    degree = degree,
    restrict_degree = restrict_degree
  )
}

bernstein.control <- function(m = 10) {
  list(m = m) # Currently only fixed order until order selection is implemented
}
