get_km <- function(object) {
  t1 <- object[, "time1"]
  t2 <- object[, "time2"]
  c <- object[, "c"]
  event1 <- object[, "event1"]
  event2 <- object[, "event2"]
  event_c <- object[, "eventc"]

  km1 <- survival::survfit(survival::Surv(t1, event1) ~ 1)
  km2 <- survival::survfit(survival::Surv(t2, event2) ~ 1)
  kmc <- survival::survfit(survival::Surv(c, event_c) ~ 1)

  return(list(km1 = km1, km2 = km2, kmc = kmc))
}
