CRFfit <- function(x, ...) {
  UseMethod("CRFfit")
}

CRFfit.default <- function(x, ...) {
  stop(
    "CRFfit() is not defined for a '",
    class(x)[1],
    "' object. Please provide a formula as input.",
    call. = FALSE
  )
}

# TODO - Add Bernstein method
CRFfit.formula <- function(
  formula,
  data,
  method = c("spline", "polynomial"),
  control = list(),
  nleqslv.control = nleqslv.control(),
  fs.control = efs.control()
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
  control <- modifyList(control.default, control)

  # Initiate and update nleqslv parameters
  nleqslv.control <- modifyList(nleqslv.control(), nleqslv.control)

  mf <- model.frame(formula = formula, data = data)
  y <- model.response(mf)

  stopifnot(inherits(y, "biSurv"))

  if (method == "bernstein") {
    stopifnot(inherits(y, "biSurvBern")) # Check whether data is suitable for Bernstein estimation
  } else {
    datalist <- prepare_data(
      y[, "time1"],
      y[, "time2"],
      y[, "event1"],
      y[, "event2"]
    )
  }

  # Fit model based on specified method
  if (method == "spline") {
    fs.control <- modifyList(efs.control(), fs.control)

    fit <- estimate_spline(
      datalist = datalist,
      spline.control = control,
      fs.control = fs.control,
      nleqslv.control = nleqslv.control
    )
    fit$model <- y
  } else if (method == "polynomial") {
    fit <- estimate_poly(
      datalist = datalist,
      nl.conrol = nleqslv.control,
      poly.control = control
    )
    fit$model <- y
  }

  return(fit)
}
