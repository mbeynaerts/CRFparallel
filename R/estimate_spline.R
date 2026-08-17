# EFS gebaseerd op de code van Simon Wood in het mgcv package (zie gam.fit4.r op github)
# gam.control() details in mgcv.r op github
estimate_spline <- function(
  y,
  # dim,
  # degree = 3,
  # lambda.init = c(1, 1),
  # start = rep(1, dim^2),
  # quantile = FALSE,
  # scale = TRUE,
  # observed.region = FALSE,
  # step.control = FALSE,
  spline.control = list(),
  efs.control = list(),
  nleqslv.control = list(),
  progress = TRUE,
  ncores = 1
) {
  # spline.control <- utils::modifyList(
  #   get("spline.control", mode = "function")(),
  #   spline.control
  # )
  # efs.control <- utils::modifyList(
  #   get("efs.control", mode = "function")(),
  #   efs.control
  # )
  # nleqslv.control <- utils::modifyList(
  #   get("nleqslv.control", mode = "function")(),
  #   nleqslv.control
  # )

  RcppParallel::setThreadOptions(numThreads = ncores)
  on.exit(RcppParallel::setThreadOptions(numThreads = 1), add = TRUE)

  repara <- FALSE # Reparameterization is currently not implemented for the extended Fellner-Schall method

  tiny <- .Machine$double.eps^0.5

  if (progress) {
    cli::cli_h1(
      "Generalized Fellner-Schall method for spline-based estimation"
    )
    cli::cli_alert("Starting now, at {Sys.time()}")
    cli::cli_progress_step("Preparing data")
  }

  datalist <- prepare_data(y, ncores = ncores)

  if (progress) {
    cli::cli_progress_step("Constructing model and penalty matrices")
  }

  obj1 <- construct_spline(
    t = datalist$X[, 1],
    delta = datalist$delta[, 1],
    dim = spline.control$dim,
    degree = spline.control$degree,
    type = spline.control$type,
    quantile = spline.control$quantile,
    scale = spline.control$scale,
    repara = repara,
    observed.region = spline.control$observed.region,
    knot.margin = spline.control$knot.margin
  )
  obj2 <- construct_spline(
    t = datalist$X[, 2],
    delta = datalist$delta[, 2],
    dim = spline.control$dim,
    degree = spline.control$degree,
    type = spline.control$type,
    quantile = spline.control$quantile,
    scale = spline.control$scale,
    repara = repara,
    observed.region = spline.control$observed.region,
    knot.margin = spline.control$knot.margin
  )

  # Save model matrix
  # final$model.matrix <- row_kron(obj1$X, obj2$X)
  # attr(final$method, "knots") <- list(knots1 = obj1$knots, knots2 = obj2$knots)

  S <- construct_tensor_penalty(obj1, obj2)
  S1 <- S[[1]]
  S2 <- S[[2]]
  ncoef <- ncol(obj1$X) * ncol(obj2$X)

  lambda.init <- spline.control$lambda.init
  lambda.new <- lambda.init # Initial lambda = 1 in Wood (2017)

  # deriv.comp <- deriv_comp(X1 = X1, X2 = X2, datalist = datalist)
  if (progress) {
    cli::cli_progress_step("Fitting model on initial values", spinner = TRUE)
  }

  fit <- efsud_fit(
    start = rep(1, ncoef),
    X1 = obj1$X,
    X2 = obj2$X,
    datalist = datalist,
    Sl = lambda.init[1] * S1 + lambda.init[2] * S2,
    control = nleqslv.control,
    ncores = ncores
  )
  k <- 1
  score <- rep(0, efs.control$maxiter)
  if (progress) {
    cli::cli_progress_step(
      msg = "Running iteration {iter}",
      msg_done = "Converged in {iter} iterations",
      msg_failed = "Failed after {iter} iterations",
      spinner = TRUE
    )
  }
  for (iter in 1:efs.control$maxiter) {
    if (progress) {
      cli::cli_progress_update()
    }
    l0 <- fit$REML

    lambda <- lambda.new

    # Some calculations to update lambda later...
    Sl <- lambda[1] * S1 + lambda[2] * S2
    Sl.inv <- MASS::ginv(Sl)

    # Update ----

    # Calculate V
    V <- solve(fit$hessian + Sl)

    # Calculate trSSj, trVS and bSb
    trSSj <- trVS <- bSb <- rep(0, length(S))
    for (i in 1:length(S)) {
      trSSj[i] <- sum(diag(Sl.inv %*% S[[i]]))
      trVS[i] <- sum(diag(V %*% S[[i]]))
      bSb[i] <- t(fit$beta) %*% S[[i]] %*% fit$beta
    }

    # Update lambdas
    a <- pmax(tiny, trSSj - trVS)
    update <- a / pmax(tiny, bSb)
    update[a == 0 & bSb == 0] <- 1
    update[!is.finite(update)] <- 1e6
    lambda.new <- pmin(update * lambda, efs.control$lambda.max)

    # Step length of update
    max.step <- max(abs(lambda.new - lambda))

    # Create new S.lambda matrix
    Sl.new <- lambda.new[1] * S1 + lambda.new[2] * S2
    # Sl.new <- lambda.new*S

    fit <- efsud_fit(
      start = fit$beta,
      X1 = obj1$X,
      X2 = obj2$X,
      datalist = datalist,
      Sl = Sl.new,
      control = nleqslv.control,
      ncores = ncores
    )
    l1 <- fit$REML

    # Start of step control ----
    if (efs.control$step.control) {
      if (l1 > l0) {
        # Improvement
        if (max.step < 1) {
          # Consider step extension
          lambda2 <- pmin(lambda * update^(k * 2), exp(12))
          fit2 <- efsud_fit(
            start = fit$beta,
            X1 = obj1$X,
            X2 = obj2$X,
            datalist = datalist,
            # Sl = lambda2*S
            Sl = lambda2[1] * S1 + lambda2[2] * S2,
            # weights = weights,
            control = nleqslv.control,
            ncores = ncores
          )
          l2 <- fit2$REML
          if (l2 > l1) {
            # Improvement - accept extension
            lambda.new <- lambda2
            l1 <- l2
            fit <- fit2
            k <- k * 2
          }
        }
      } else {
        # No improvement
        lk <- l1
        lambda3 <- lambda.new
        while (lk < l0 && k > 1) {
          # Don't contract too much since the likelihood does not need to increase k > 0.001
          k <- k / 2 ## Contract step
          lambda3 <- pmin(lambda * update^k, efs.control$lambda.max)
          fit <- efsud_fit(
            start = fit$beta,
            X1 = obj1$X,
            X2 = obj2$X,
            datalist = datalist,
            # Sl = lambda3*S
            Sl = lambda3[1] * S1 + lambda3[2] * S2,
            # weights = weights,
            control = nleqslv.control,
            ncores = ncores
          )
          lk <- fit$REML
        }
        lambda.new <- lambda3
        l1 <- lk
        max.step <- max(abs(lambda.new - lambda))
        if (k < 1) k <- 1
      }
    } # end of step length control

    # save loglikelihood value
    score[iter] <- l1

    # Break procedures ----

    # Break procedure if REML change and step size are too small
    if (
      iter > 3 &&
        max(abs(diff(score[(iter - 3):iter]))) < efs.control$REML.tol &&
        max.step < efs.control$lambda.tol
    ) {
      if (progress) {
        cli::cli_alert_info("REML not changing")
      }
      break
    }
    # Or break is likelihood does not change
    # if (l1 == l0) {if (progress) print("Loglik not changing"); break}

    # Stop if loglik is not changing
    if (iter == 1) {
      old.ll <- fit$ll
    } else {
      if (abs(old.ll - fit$ll) < efs.control$ll.tol) {
        if (progress) {
          cli::cli_alert_info("Log-likelihood not changing")
        }
        break
      } # if (abs(old.ll-fit$ll)<100*eps*abs(fit$ll))
      old.ll <- fit$ll
    }

    cli::cli_progress_update() # To make sure that the spinner in cli_progress_step works
  } # End of for loop

  # final$coefficients <- fit$beta
  # final$fitted.values <- as.vector(final$model.matrix %*% fit$beta)
  # final$vcov <- 2 * solve(fit$hessian + lambda.new[1] * S1 + lambda.new[2] * S2)
  # attr(final$method, "lambda") <- lambda.new
  # attr(final$method, "iterations") <- iter
  # final$loglik <- fit$ll
  # final$reml <- fit$REML

  if (progress) {
    cli::cli_progress_step("Creating output")
  }

  final <- CRFspline(
    model.matrix = row_kron(obj1$X, obj2$X),
    idx = seq_len(ncoef),
    splines2_list = list(X1 = obj1$X, X2 = obj2$X),
    vcov = 2 * solve(fit$hessian + lambda.new[1] * S1 + lambda.new[2] * S2),
    lambda = lambda.new,
    coefficients = fit$beta,
    loglik = fit$ll,
    reml = fit$REML,
    iterations = iter,
    call = match.call()
  )

  if (progress) {
    cli::cli_alert("Finished at {Sys.time()}")
  }

  return(final)

  # return(list(
  #   beta = fit$beta,
  #   lambda = lambda.new,
  #   vcov = 2 * solve(fit$hessian + lambda.new[1] * S1 + lambda.new[2] * S2),
  #   iterations = iter,
  #   ll = fit$ll,
  #   history = score[1:iter],
  #   info = fit$estim,
  #   splinepar = list(dim = dim, degree = degree, XP1 = obj1$XP, XP2 = obj2$XP),
  #   knots = list(knots1 = obj1$knots, knots2 = obj2$knots)
  # ))
}
