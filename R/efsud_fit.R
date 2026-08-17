efsud_fit <- function(
  start,
  X1,
  X2,
  datalist,
  Sl,
  control = nleqslv.control(),
  ncores = 1
) {
  # if (is.null(deriv.comp)) deriv <- deriv_comp(X1 = X1, X2 = X2, datalist = datalist, weights = weights)
  # else deriv <- deriv.comp

  # beta <- multiroot(gradient_spline, start = start, jacfunc = hessian_spline, jactype = "fullusr", rtol = 1e-10, X1 = X1, X2 = X2, Sl = Sl, datalist = datalist, deriv = deriv)$root

  estim <- nleqslv::nleqslv(
    x = start,
    fn = gradient_spline,
    jac = hessian_spline,
    method = control$method,
    global = control$global,
    control = control[setdiff(names(control), c("method", "global"))],
    X1 = X1,
    X2 = X2,
    datalist = datalist,
    Sl = Sl,
    ncores = ncores
  )
  beta <- estim$x
  if (any(is.na(beta))) {
    estim
    cli::cli_abort("One of the spline coefficients is NA")
  }

  H <- hessian_spline(
    coef.vector = beta,
    X1 = X1,
    X2 = X2,
    datalist = datalist,
    ncores = ncores
  )

  fit <- reml_spline(
    coef.vector = beta,
    X1 = X1,
    X2 = X2,
    Sl = Sl,
    H = H,
    minusLogLik = FALSE,
    datalist = datalist
  )

  return(list(
    beta = beta,
    hessian = H,
    REML = fit$REML,
    ll = fit$ll,
    info = estim
  ))
}
