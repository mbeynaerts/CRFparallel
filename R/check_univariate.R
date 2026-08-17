check_univariate <- function(time1, time2, event1, event2) {
  # NOTE: The input variables are not checked for class or type, as this is done in the biSurv constructor. This function only checks whether the censoring mechanism is univariate, which is required for the Bernstein method.

  bernstein <- identical(
    (event1 == 0 & event2 == 0),
    (time1 == time2)
  )
  return(bernstein)
}
