estimate_bernstein <- function(
  data,
  bernstein.control,
  progress,
  ncores
) {
  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1), add = TRUE)

  if (progress) {
    cli::cli_h1(
      "Bernstein-based estimation"
    )
    cli::cli_alert("Starting now, at {Sys.time()}")
    cli::cli_progress_step("Preparing data")
  }

  km_list <- get_km(data) # Marginal KM-estimates
  tau1 <- min(km_list$km1$surv)
  tau2 <- min(km_list$km2$surv)
  max_c <- max(data[, "timec"])

  if (progress) {
    cli::cli_progress_step("Estimating empirical copula")
  }

  copula <- estimate_copula(
    t1 = data[, "time1"],
    t2 = data[, "time2"],
    max_c = max_c,
    km1 = km_list$km1,
    km2 = km_list$km2,
    kmc = km_list$kmc,
    m = bernstein.control$m,
    tau1 = tau1,
    tau2 = tau2
  )

  if (progress) {
    cli::cli_progress_step("Estimating CRF using Bernstein estimator")
  }

  fitted.values <- bernstein_estimator_vec(
    s1 = km_list$km1$surv,
    s2 = km_list$km2$surv,
    m = bernstein.control$m,
    tau1 = tau1,
    tau2 = tau2
  )

  if (progress) {
    cli::cli_progress_step("Creating output")
  }

  final <- CRFbern(
    method.args = bernstein.control,
    tau = c(tau1, tau2),
    marginal_km = km_list,
    call = match.call()
  )

  if (progress) {
    cli::cli_alert("Finished at {Sys.time()}")
  }

  return(final)
}
