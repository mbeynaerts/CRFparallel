biSurv <- function(time1, time2, event1, event2 = NULL) {
  if (any(missing(time1), missing(time2))) {
    stop("Must have a time argument.")
  }

  if (inherits(time1, "difftime")) {
    time1 <- unclass(time1)
  }
  if (inherits(time2, "difftime")) {
    time2 <- unclass(time2)
  }
  if (!is.numeric(time1) | !is.numeric(time2)) {
    stop("At least one time variable is not numeric.")
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
    stop("Time and/or event variables cannot contain NA values.")
  }

  # check that all variables have the same length
  if (
    !identical(length(time1), length(event1), length(time2), length(event2))
  ) {
    stop("Time and event variables must have the same length.")
  }

  # Convert all possible event formats to 1/0 (observed/censored)
  if (is.logical(event1)) {
    event1 <- as.integer(event1)
  } else if (is.factor(event1)) {
    if (length(levels(event1)) > 2) {
      stop("Event variable must have at most 2 levels.")
    }
    event1 <- as.integer(event1) - 1
  } else if (is.numeric(event1)) {
    # Check whether event is 1/0 coded
    if (length(unique(event1)) > 2) {
      stop("Numeric event variable must be binary (0/1).")
    }
    event1[event1 > 1] <- 1
  } else {
    stop("Event variable must be numeric, logical or factor.")
  }

  if (is.logical(event2)) {
    event2 <- as.integer(event2)
  } else if (is.factor(event2)) {
    if (length(levels(event2)) > 2) {
      stop("Event variable must have at most 2 levels.")
    }
    event2 <- as.integer(event2) - 1
  } else if (is.numeric(event2)) {
    # Check whether event is 1/0 coded
    if (length(unique(event2)) > 2) {
      stop("Numeric event variable must be binary (0/1).")
    }
    event2[event2 > 1] <- 1
  } else {
    stop("Event variable must be numeric, logical or factor.")
  }

  ss <- data.frame(time1, time2, event1, event2)

  # Check for univariate censoring to allow for bernstein estimation
  bernstein <- identical(
    (event1 == 0 & event2 == 0),
    (time1 == time2)
  )

  if (bernstein) {
    ss$c <- pmax(ss$time1, ss$time2)
    ss$eventc <- 1 - event1 * event2
  }

  class(ss) <- "biSurv"
  attr(ss, "bernstein") <- bernstein

  return(ss)
}
