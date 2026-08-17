construct_tensor_penalty <- function(object1, object2) {
  # See Reiss et al. (2014) for details on the construction of the penalty matrices
  df1 <- ncol(object1$X)
  df2 <- ncol(object2$X)

  S1 <- object1$S %x% diag(df2)
  S2 <- diag(df1) %x% object2$S

  return(list(S1 = S1, S2 = S2))
}
