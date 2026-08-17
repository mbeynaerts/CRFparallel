#' @export
biSurv <- S7::new_class(
  "biSurv",
  properties = list(
    data = new_property(class_numeric, validator = function(value) {
      if (!inherits(value, "matrix")) "@data must be a matrix."
    }),
    univariate = new_property(class_logical, default = FALSE)
  ),
  constructor = function(time1, time2, event1, event2, data) {
    # Check whether the variables of interest are part of 'data' argument
    if (!missing(data)) {
      env <- parent.frame()
      time1 <- eval(quote(time1), envir = data, enclos = env)
      time2 <- eval(quote(time2), envir = data, enclos = env)
      event1 <- eval(quote(event1), envir = data, enclos = env)
      event2 <- eval(quote(event2), envir = data, enclos = env)
    }

    if (any(missing(time1), missing(time2))) {
      cli::cli_abort("Must have a time argument.")
    }

    if (inherits(time1, "difftime")) {
      time1 <- unclass(time1)
    }
    if (inherits(time2, "difftime")) {
      time2 <- unclass(time2)
    }
    if (!is.numeric(time1) | !is.numeric(time2)) {
      cli::cli_abort("At least one time variable is not numeric.")
    }

    # If event is missing, assume all events are observed (i.e., no censoring)
    if (missing(event1)) {
      event1 <- rep(1, length(time1))
    }
    if (missing(event2)) {
      event2 <- rep(1, length(time2))
    }

    # Check for NA values
    if (
      any(is.na(time1)) |
        any(is.na(time2)) |
        any(is.na(event1)) |
        any(is.na(event2))
    ) {
      cli::cli_abort("Time and/or event variables cannot contain NA values.")
    }

    # check that all variables have the same length
    if (
      !identical(length(time1), length(event1), length(time2), length(event2))
    ) {
      cli::cli_abort("Time and event variables must have the same length.")
    }

    # Convert all possible event formats to 1/0 (observed/censored)
    if (is.logical(event1)) {
      event1 <- as.integer(event1)
    } else if (is.factor(event1)) {
      if (length(levels(event1)) > 2) {
        cli::cli_abort("Event variable must have at most 2 levels.")
      }
      event1 <- as.integer(event1) - 1
    } else if (is.numeric(event1)) {
      # Check whether event is 1/0 coded
      if (length(unique(event1)) > 2) {
        cli::cli_abort("Numeric event variable must be binary (0/1).")
      }
      event1[event1 > 1] <- 1
    } else {
      cli::cli_abort("Event variable must be numeric, logical or factor.")
    }

    if (is.logical(event2)) {
      event2 <- as.integer(event2)
    } else if (is.factor(event2)) {
      if (length(levels(event2)) > 2) {
        cli::cli_abort("Event variable must have at most 2 levels.")
      }
      event2 <- as.integer(event2) - 1
    } else if (is.numeric(event2)) {
      # Check whether event is 1/0 coded
      if (length(unique(event2)) > 2) {
        cli::cli_abort("Numeric event variable must be binary (0/1).")
      }
      event2[event2 > 1] <- 1
    } else {
      cli::cli_abort("Event variable must be numeric, logical or factor.")
    }

    # Check for univariate censoring to allow bernstein method
    bernstein <- check_univariate(time1, time2, event1, event2)

    if (bernstein) {
      ss <- cbind(
        time1,
        time2,
        event1,
        event2,
        pmax(time1, time2),
        1 - event1 * event2
      )
      colnames(ss) <- c("time1", "time2", "event1", "event2", "timec", "eventc")
    } else {
      ss <- cbind(time1, time2, event1, event2)
      colnames(ss) <- c("time1", "time2", "event1", "event2")
    }

    new_object(S7_object(), data = ss, univariate = bernstein)
  }
)

#' @export
is.biSurv <- S7::new_generic("is.biSurv", "x")
S7::method(is.biSurv, biSurv) <- function(x) {
  S7_inherits(x, "biSurv")
}

as.data.frame <- S7::new_external_generic("base", "as.data.frame", "x")
S7::method(as.data.frame, biSurv) <- function(
  x,
  row.names = NULL,
  optional = FALSE,
  make.names = TRUE,
  ...,
  stringsAsFactors = FALSE
) {
  # @data is validated in the biSurv constructor
  if (!inherits(x@data, "matrix")) {
    return(as.data.frame.default(x, ...))
  }
  as.data.frame.matrix(
    x@data,
    row.names = NULL,
    optional = FALSE,
    make.names = TRUE,
    ...,
    stringsAsFactors = FALSE
  )
}

# #' Create a biSurv object for bivariate survival data
# #'
# #' @param time1 A numeric vector
# #' @param time2 A numeric vector
# #' @param event1 A vector
# #' @param event2 A vector
# #'
# #' @returns An object of class {.cls biSurv}
# #'
# #' @export
# #' @examples
# biSurv <- function(time1, time2, event1, event2) {
#   if (any(missing(time1), missing(time2))) {
#     cli::cli_abort("Must have a time argument.")
#   }

#   if (inherits(time1, "difftime")) {
#     time1 <- unclass(time1)
#   }
#   if (inherits(time2, "difftime")) {
#     time2 <- unclass(time2)
#   }
#   if (!is.numeric(time1) | !is.numeric(time2)) {
#     cli::cli_abort("At least one time variable is not numeric.")
#   }

#   # If event is missing, assume all events are observed (i.e., no censoring)
#   if (missing(event1)) {
#     event1 <- rep(1, length(time1))
#   }
#   if (missing(event2)) {
#     event2 <- rep(1, length(time2))
#   }

#   # Check for NA values
#   if (
#     any(is.na(time1)) |
#       any(is.na(time2)) |
#       any(is.na(event1)) |
#       any(is.na(event2))
#   ) {
#     cli::cli_abort("Time and/or event variables cannot contain NA values.")
#   }

#   # check that all variables have the same length
#   if (
#     !identical(length(time1), length(event1), length(time2), length(event2))
#   ) {
#     cli::cli_abort("Time and event variables must have the same length.")
#   }

#   # Convert all possible event formats to 1/0 (observed/censored)
#   if (is.logical(event1)) {
#     event1 <- as.integer(event1)
#   } else if (is.factor(event1)) {
#     if (length(levels(event1)) > 2) {
#       cli::cli_abort("Event variable must have at most 2 levels.")
#     }
#     event1 <- as.integer(event1) - 1
#   } else if (is.numeric(event1)) {
#     # Check whether event is 1/0 coded
#     if (length(unique(event1)) > 2) {
#       cli::cli_abort("Numeric event variable must be binary (0/1).")
#     }
#     event1[event1 > 1] <- 1
#   } else {
#     cli::cli_abort("Event variable must be numeric, logical or factor.")
#   }

#   if (is.logical(event2)) {
#     event2 <- as.integer(event2)
#   } else if (is.factor(event2)) {
#     if (length(levels(event2)) > 2) {
#       cli::cli_abort("Event variable must have at most 2 levels.")
#     }
#     event2 <- as.integer(event2) - 1
#   } else if (is.numeric(event2)) {
#     # Check whether event is 1/0 coded
#     if (length(unique(event2)) > 2) {
#       cli::cli_abort("Numeric event variable must be binary (0/1).")
#     }
#     event2[event2 > 1] <- 1
#   } else {
#     cli::cli_abort("Event variable must be numeric, logical or factor.")
#   }

#   # Check for univariate censoring to allow bernstein method
#   bernstein <- check_univariate(time1, event1, time2, event2)

#   if (bernstein) {
#     ss <- cbind(
#       time1,
#       time2,
#       event1,
#       event2,
#       pmax(time1, time2),
#       1 - event1 * event2
#     )
#     colnames(ss) <- c("time1", "time2", "event1", "event2", "timec", "eventc")
#     class(ss) <- c("biSurvUniv", "biSurv", "matrix")
#   } else {
#     ss <- cbind(time1, time2, event1, event2)
#     colnames(ss) <- c("time1", "time2", "event1", "event2")
#     class(ss) <- c("biSurv", "matrix")
#   }

#   return(ss)
# }

# as.biSurv <- function(y) {
#   if (!inherits(y, "biSurv")) {
#     y <- as.matrix(y)
#     class(y) <- c("biSurv", "matrix")
#   }

#   # restore column names if lost
#   if (is.null(colnames(y)) || ncol(y) == 4) {
#     colnames(y) <- c("time1", "time2", "event1", "event2")
#   }

#   # recompute bernstein (robust!)
#   attr(y, "bernstein") <- identical(
#     (y[, 3] == 0 & y[, 4] == 0),
#     (y[, 1] == y[, 2])
#   )

#   y
# }

# #' @export
# "[.biSurv" <- function(x, i, j, drop = FALSE) {
#   res <- NextMethod("[")
#   class(res) <- class(x)
#   res
# }

# #' @export
# "[.biSurvUniv" <- function(x, i, j, drop = FALSE) {
#   res <- NextMethod("[")
#   class(res) <- class(x)
#   res
# }

# #' @export
# is.biSurv <- function(x) {
#   inherits(x, "biSurv")
# }

# #' @export
# as.data.frame.biSurv <- function(x, row.names = NULL, optional = FALSE, ...) {
#   df <- as.data.frame(
#     unclass(x),
#     row.names = row.names,
#     optional = optional,
#     ...
#   )
#   class(df) <- setdiff(class(x), c("biSurv", "matrix"))
#   return(df)
# }

# #' @export
# as.data.frame.biSurvUniv <- function(
#   x,
#   row.names = NULL,
#   optional = FALSE,
#   ...
# ) {
#   df <- NextMethod()
#   class(df) <- setdiff(class(x), c("biSurvUniv", "biSurv", "matrix"))
#   return(df)
# }
