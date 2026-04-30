get_km <- function(object) {
  stopifnot(inherits(object, "biSurv"))
  stopifnot(isTRUE(attr(object, "bernstein")))

  t1 <- object$t1
  t2 <- object$t2
  event1 <- object$event1
  event2 <- object$event2

  # Calculate the censoring time and event indicator for the combined censoring variable
  c <- pmax(t1, t2)
  event_c <- 1 - event1 * event2

  km1 <- survival::survfit(survival::Surv(t1, event1) ~ 1)
  km2 <- survival::survfit(survival::Surv(t2, event2) ~ 1)
  kmc <- survival::survfit(survival::Surv(c, event_c) ~ 1)

  return(list(km1 = km1, km2 = km2, kmc = kmc))
}
