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
    "polynomial" = polynomial.control()
  )
  control <- modifyList(control.default, control)

  # Initiate and update nleqslv parameters
  nleqslv.control <- modifyList(nleqslv.control(), nleqslv.control)

  data.parsed <- model.frame(formula = formula, data = data)
  datalist <- prepare_data(
    data.parsed$time1,
    data.parsed$time2,
    data.parsed$event1,
    data.parsed$event2
  )

  # Fit model based on specified method
  if (method == "spline") {
    fs.control <- modifyList(efs.control(), fs.control)

    fit <- estimate_spline(
      datalist = datalist,
      spline.control = control,
      fs.control = fs.control,
      nleqslv.control = nleqslv.control
    )
    fit$model <- data.parsed
  } else if (method == "polynomial") {
    fit <- estimate_poly(
      datalist = datalist,
      nl.conrol = nleqslv.control,
      poly.control = control
    )
    fit$model <- data.parsed
  }

  return(fit)
}
