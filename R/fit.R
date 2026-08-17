#' @export
fit <- S7::new_generic("fit", "biSurv")

#' Fit the cross ratio functions using splines, polynomials or Bernstein polynomials.
#'
#' @param biSurv An object of class `biSurv`.
#' @param method A character string specifying the method to be used for fitting the model. Options are 'spline' for spline-based estimation, 'polynomial' for polynomial-based estimation, and 'bernstein' for Bernstein polynomial-based estimation. Default is 'spline'.
#' @param control An optional list of control parameters for the `nleqslv` optimization algorithm. See [spline.control()] for spline-based estimation, [polynomial.control()] for polynomial-based estimation, and [bernstein.control()] for Bernstein-based estimation.
#' @param nleqslv.control An optional list of control parameters for the `nleqslv` optimization algorithm. See [nleqslv.control()] for details.
#' @param fs.control An optional list of control parameters for the generalized Fellner-Schall method. See [efs.control()] for details.
#' @param ncores An integer specifying the number of cores to use for multi-threading. Default is 1.
#' @param progress Logical indicating whether to print progress messages during model fitting. Default is `TRUE`.
#'
#' @returns An object of class `CRFspline`, `CRFpoly` or `CRFbern` containing the fitted model parametersn and other relevant information depending on the method used for estimation.
#'
#' @export
method(fit, biSurv) <- function(
  biSurv, # Should be of the form biSurv(time1, time2, event1, event2)
  method = c("polynomial", "spline", "bernstein"),
  control = list(),
  nleqslv.control = list(),
  fs.control = list(),
  ncores = 1,
  progress = TRUE
) {
  # Check that method is valid
  method <- match.arg(method)

  # Initiate and update control parameters
  control.default <- switch(
    method,
    "spline" = spline.control(),
    "polynomial" = polynomial.control(),
    "bernstein" = bernstein.control()
  )

  control.new <- utils::modifyList(control.default, control)

  # Initiate and update nleqslv parameters
  nleqslv.default <- nleqslv.control()
  nleqslv.control.new <- utils::modifyList(nleqslv.default, nleqslv.control)

  # mf <- model.frame(formula = formula, data = data)
  # y <- model.response(mf)

  if (method == "bernstein" & isFALSE(biSurv@univariate)) {
    cli::cli_abort(
      "Method 'Bernstein' is only available for univariate censored data."
    )
  }

  # Fit model based on specified method
  if (method == "spline") {
    efs.control.default <- efs.control()
    efs.control.new <- utils::modifyList(efs.control.default, fs.control)
    fit <- estimate_spline(
      y = biSurv@data,
      spline.control = control.new,
      efs.control = efs.control.new,
      nleqslv.control = nleqslv.control.new,
      progress = progress,
      ncores = ncores
    )
  } else if (method == "polynomial") {
    fit <- estimate_poly(
      y = biSurv@data,
      nleqslv.control = nleqslv.control.new,
      poly.control = control.new,
      progress = progress,
      ncores = ncores
    )
  } else if (method == "bernstein") {
    fit <- estimate_bernstein(
      data = biSurv@data,
      bernstein.control = control.new,
      progress = progress,
      ncores = ncores
    )
  }

  return(fit)
}
