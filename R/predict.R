predict.CRFpoly <- function(
  object,
  new_data,
  se.fit = FALSE,
  interval = c("none", "confidence"),
  level = 0.95,
  log = TRUE,
  ...
) {
  if (!inherits(object, "CRFpoly")) {
    stop("Object must be of class 'CRFpoly'.")
  }

  d <- object$degree
  beta <- rep(0, (d + 1)^2)
  beta[attr(object$model.matrix, "idx")] <- object$coefficients

  if (missing(new_data) || is.null(new_data)) {
    pred <- object$fitted.values
    X <- object$model.matrix
  } else {
    if (!is.data.frame(new_data)) {
      stop("New data must be a data frame.")
    }
    if (ncol(new_data) != 2) {
      stop(
        "New data must have exactly two columns corresponding to two event times."
      )
    }
    if (!all(sapply(new_data, is.numeric))) {
      stop("All columns in new data must be numeric.")
    }
    X1 <- model.matrix(
      ~ poly(new_data[[1]], degree = d, raw = TRUE, simple = TRUE)
    )
    X2 <- model.matrix(
      ~ poly(new_data[[2]], degree = d, raw = TRUE, simple = TRUE)
    )
    X <- row_kron(X1, X2)
    pred <- X %*% object$coefficients
  }

  if (!log) {
    out <- data.frame(.pred = exp(pred))
  } else {
    out <- data.frame(.pred = pred)
  }

  if (se.fit || interval != "none") {
    se.logtheta <- sqrt(rowSums((X %*% object$vcov) * X))
  }

  if (se.fit) {
    out$.std_error <- se.logtheta
    if (!log) {
      out$.std_error <- exp(pred) * se.logtheta
    }
  }

  interval <- match.arg(interval)
  if (interval == "confidence") {
    z <- qnorm(1 - (1 - level) / 2)
    lower <- pred - z * se.logtheta
    upper <- pred + z * se.logtheta
    if (!log) {
      out$.pred_lower <- exp(lower)
      out$.pred_upper <- exp(upper)
    } else {
      out$.pred_lower <- lower
      out$.pred_upper <- upper
    }
  }

  return(out)
}

predict.CRFspline <- function(
  object,
  new_data,
  se.fit = FALSE,
  interval = c("none", "confidence"),
  level = 0.95,
  log = TRUE,
  ...
) {
  if (!inherits(object, "CRFspline")) {
    stop("Object must be of class 'CRFspline'.")
  }

  if (missing(new_data) || is.null(new_data)) {
    pred <- object$fitted.values
    X <- object$model.matrix
  } else {
    if (!is.data.frame(new_data)) {
      stop("New data must be a data frame.")
    }
    if (ncol(new_data) != 2) {
      stop(
        "New data must have exactly two columns corresponding to two event times."
      )
    }
    if (!all(sapply(new_data, is.numeric))) {
      stop("All columns in new data must be numeric.")
    }
    X1 <- extrapolate_spline(object, new_data[[1]], index = 1)
    X2 <- extrapolate_spline(object, new_data[[2]], index = 2)
    X <- row_kron(X1, X2)
    pred <- X %*% object$coefficients
  }

  if (!log) {
    out <- data.frame(.pred = exp(pred))
  } else {
    out <- data.frame(.pred = pred)
  }

  if (se.fit || interval != "none") {
    se.logtheta <- sqrt(rowSums((X %*% object$vcov) * X))
  }

  if (se.fit) {
    out$.std_error <- se.logtheta
    if (!log) {
      out$.std_error <- exp(pred) * se.logtheta
    }
  }

  interval <- match.arg(interval)
  if (interval == "confidence") {
    z <- qnorm(1 - (1 - level) / 2)
    lower <- pred - z * se.logtheta
    upper <- pred + z * se.logtheta
    if (!log) {
      out$.pred_lower <- exp(lower)
      out$.pred_upper <- exp(upper)
    } else {
      out$.pred_lower <- lower
      out$.pred_upper <- upper
    }
  }

  return(out)
}
