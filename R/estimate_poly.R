estimate_poly <- function(
  y,
  start = rep(0, 10),
  degree = 3,
  restrict_degree = 3,
  nleqslv.control = list(),
  poly.control = list(),
  progress = TRUE,
  ncores = 1
) {
  # nleqslv.control <- utils::modifyList(
  #   get("nleqslv.control", mode = "function")(),
  #   nleqslv.control
  # )
  # poly.control <- utils::modifyList(
  #   get("polynomial.control", mode = "function")(),
  #   poly.control
  # )

  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1), add = TRUE)

  if (progress) {
    cli::cli_h1(
      "Maximum likelihood for polynomial-based estimation"
    )
    cli::cli_alert("Starting now, at {Sys.time()}")
    cli::cli_progress_step("Preparing data")
  }

  datalist <- prepare_data(y, ncores = ncores)

  # X1 <- cbind(1, datalist$X[, 1], datalist$X[, 1]^2, datalist$X[, 1]^3)
  # X2 <- cbind(1, datalist$X[, 2], datalist$X[, 2]^2, datalist$X[, 2]^3)

  if (progress) {
    cli::cli_progress_step("Constructing model matrix")
  }

  X1 <- model.matrix(
    ~ poly(datalist$X[, 1], degree = degree, raw = TRUE, simple = TRUE)
  )
  X2 <- model.matrix(
    ~ poly(datalist$X[, 2], degree = degree, raw = TRUE, simple = TRUE)
  )

  model.matrix <- row_kron(X1, X2)

  idx <- keep_indices(
    poly_degree = degree,
    restrict_degree = restrict_degree
  )

  if (progress) {
    cli::cli_progress_step("Estimating model parameters")
  }

  beta <- nleqslv::nleqslv(
    #FIXME Add ... output from nleqslv.control to input of nleqslv
    x = start,
    fn = gradient_poly,
    jac = hessian_poly,
    method = nleqslv.control$method,
    global = nleqslv.control$global,
    idx = idx,
    datalist = datalist,
    X1 = X1,
    X2 = X2
  )

  if (progress) {
    cli::cli_progress_step("Calculating variance-covariance matrix")
  }

  V <- hessian_poly(beta$x, X1 = X1, X2 = X2, datalist = datalist, idx = idx)

  # final$fitted.values <- X1 %*% coef.matrix %*% t(X2)
  # final$coefficients <- beta$x
  # final$vcov <- solve(V)
  # final$loglik <- beta$fvec

  if (progress) {
    cli::cli_progress_step("Creating output")
  }

  final <- CRFpoly(
    poly.control,
    model.matrix = model.matrix,
    idx = idx + 1, # Convert C++ idx to R idx
    vcov = solve(V),
    coefficients = beta$x,
    loglik = beta$fvec,
    call = match.call()
  )

  if (progress) {
    cli::cli_alert("Finished at {Sys.time()}")
  }

  return(final)
}
