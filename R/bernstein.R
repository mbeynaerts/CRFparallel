FastComputeCRF2 = function(s1, s2, m, tau1, tau2, Chat) {
  sum1 = 0
  sum2 = 0
  sum3 = 0
  sum4 = 0
  w1 = (s1 - tau1) / (1 - tau1)
  w2 = (s2 - tau2) / (1 - tau2)
  for (k in 0:m) {
    for (l in 0:m) {
      pmk = choose(m, k) * w1^(k) * (1 - w1)^(m - k)
      pml = choose(m, l) * w2^(l) * (1 - w2)^(m - l)
      pmk_acc = pmk * (k - m * w1) / (w1 * (1 - w1))
      pml_acc = pml * (l - m * w2) / (w2 * (1 - w2))
      chat = Chat[as.character(k), as.character(l)]
      sum1 = sum1 + chat * pmk * pml
      sum2 = sum2 + chat * pmk_acc * pml
      sum3 = sum3 + chat * pmk * pml_acc
      sum4 = sum4 + chat * pmk_acc * pml_acc
    }
  }
  Cm = sum1
  C12m = sum4
  C1m = sum2
  C2m = sum3
  thetam = sum1 * sum4 / (sum2 * sum3)
  # return(list(thetam = thetam, Cm = Cm, C12m = C12m, C1m = C1m, C2m = C2m))
  return(thetam)
}
