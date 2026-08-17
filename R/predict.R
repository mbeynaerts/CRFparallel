predict <- S7::new_external_generic("stats", "predict", "object")

#' @export
S7::method(predict, CRFpoly) <- function(
  object,
  new_data = NULL,
  se.fit = FALSE,
  interval = c("none", "confidence"),
  level = 0.95,
  log = TRUE
) {
  d <- object@method@degree
  coef <- object@coefficients@coefficients
  beta <- rep(0, (d + 1)^2)
  beta[object@model.matrix@idx] <- coef

  if (missing(new_data) || is.null(new_data)) {
    pred <- object@fitted.values
    X <- object@model.matrix@model.matrix
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
    pred <- X %*% coef
  }

  if (!log) {
    out <- data.frame(.pred = exp(pred))
  } else {
    out <- data.frame(.pred = pred)
  }

  if (se.fit || interval != "none") {
    se.logtheta <- sqrt(rowSums((X %*% object@vcov) * X))
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

#' @export
S7::method(predict, CRFspline) <- function(
  object,
  new_data = NULL,
  se.fit = FALSE,
  interval = c("none", "confidence"),
  level = 0.95,
  log = TRUE
) {
  if (missing(new_data) || is.null(new_data)) {
    pred <- object@fitted.values
    X <- object@model.matrix@model.matrix
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
    splines2_x <- object@splines2[[1]]
    splines2_y <- object@splines2[[2]]
    X1 <- splines2::naturalSpline(
      new_data[[1]],
      degree = attr(splines2_x, "degree"),
      intercept = TRUE,
      knots = attr(splines2_x, "knots"),
      Boundary.knots = attr(splines2_x, "Boundary.knots")
    )
    X2 <- splines2::naturalSpline(
      new_data[[2]],
      degree = attr(splines2_y, "degree"),
      intercept = TRUE,
      knots = attr(splines2_y, "knots"),
      Boundary.knots = attr(splines2_y, "Boundary.knots")
    )
    # X1 <- extrapolate_spline(object, new_data[[1]], index = 1)
    # X2 <- extrapolate_spline(object, new_data[[2]], index = 2)
    X <- row_kron(X1, X2)
    pred <- X %*% object@coefficients@coefficients
  }

  if (!log) {
    out <- data.frame(.pred = exp(pred))
  } else {
    out <- data.frame(.pred = pred)
  }

  if (se.fit || interval != "none") {
    se.logtheta <- sqrt(rowSums((X %*% object@vcov) * X))
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

# #' @export
# predict.CRFspline <- function(
#   object,
#   new_data,
#   se.fit = FALSE,
#   interval = c("none", "confidence"),
#   level = 0.95,
#   log = TRUE,
#   ...
# ) {
#   if (!inherits(object, "CRFspline")) {
#     stop("Object must be of class 'CRFspline'.")
#   }

#   if (missing(new_data) || is.null(new_data)) {
#     pred <- object$fitted.values
#     X <- object$model.matrix
#   } else {
#     if (!is.data.frame(new_data)) {
#       stop("New data must be a data frame.")
#     }
#     if (ncol(new_data) != 2) {
#       stop(
#         "New data must have exactly two columns corresponding to two event times."
#       )
#     }
#     if (!all(sapply(new_data, is.numeric))) {
#       stop("All columns in new data must be numeric.")
#     }
#     X1 <- extrapolate_spline(object, new_data[[1]], index = 1)
#     X2 <- extrapolate_spline(object, new_data[[2]], index = 2)
#     X <- row_kron(X1, X2)
#     pred <- X %*% object$coefficients
#   }

#   if (!log) {
#     out <- data.frame(.pred = exp(pred))
#   } else {
#     out <- data.frame(.pred = pred)
#   }

#   if (se.fit || interval != "none") {
#     se.logtheta <- sqrt(rowSums((X %*% object$vcov) * X))
#   }

#   if (se.fit) {
#     out$.std_error <- se.logtheta
#     if (!log) {
#       out$.std_error <- exp(pred) * se.logtheta
#     }
#   }

#   interval <- match.arg(interval)
#   if (interval == "confidence") {
#     z <- qnorm(1 - (1 - level) / 2)
#     lower <- pred - z * se.logtheta
#     upper <- pred + z * se.logtheta
#     if (!log) {
#       out$.pred_lower <- exp(lower)
#       out$.pred_upper <- exp(upper)
#     } else {
#       out$.pred_lower <- lower
#       out$.pred_upper <- upper
#     }
#   }

#   return(out)
# }

# #' @export
# predict.CRFpoly <- function(
#   object,
#   new_data,
#   se.fit = FALSE,
#   interval = c("none", "confidence"),
#   level = 0.95,
#   log = TRUE,
#   ...
# ) {
#   if (!inherits(object, "CRFpoly")) {
#     stop("Object must be of class 'CRFpoly'.")
#   }

#   d <- object$degree
#   beta <- rep(0, (d + 1)^2)
#   beta[attr(object$model.matrix, "idx")] <- object$coefficients

#   if (missing(new_data) || is.null(new_data)) {
#     pred <- object$fitted.values
#     X <- object$model.matrix
#   } else {
#     if (!is.data.frame(new_data)) {
#       stop("New data must be a data frame.")
#     }
#     if (ncol(new_data) != 2) {
#       stop(
#         "New data must have exactly two columns corresponding to two event times."
#       )
#     }
#     if (!all(sapply(new_data, is.numeric))) {
#       stop("All columns in new data must be numeric.")
#     }
#     X1 <- model.matrix(
#       ~ poly(new_data[[1]], degree = d, raw = TRUE, simple = TRUE)
#     )
#     X2 <- model.matrix(
#       ~ poly(new_data[[2]], degree = d, raw = TRUE, simple = TRUE)
#     )
#     X <- row_kron(X1, X2)
#     pred <- X %*% object$coefficients
#   }

#   if (!log) {
#     out <- data.frame(.pred = exp(pred))
#   } else {
#     out <- data.frame(.pred = pred)
#   }

#   if (se.fit || interval != "none") {
#     se.logtheta <- sqrt(rowSums((X %*% object$vcov) * X))
#   }

#   if (se.fit) {
#     out$.std_error <- se.logtheta
#     if (!log) {
#       out$.std_error <- exp(pred) * se.logtheta
#     }
#   }

#   interval <- match.arg(interval)
#   if (interval == "confidence") {
#     z <- qnorm(1 - (1 - level) / 2)
#     lower <- pred - z * se.logtheta
#     upper <- pred + z * se.logtheta
#     if (!log) {
#       out$.pred_lower <- exp(lower)
#       out$.pred_upper <- exp(upper)
#     } else {
#       out$.pred_lower <- lower
#       out$.pred_upper <- upper
#     }
#   }

#   return(out)
# }
