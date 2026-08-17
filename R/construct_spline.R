construct_spline <- function(
  t,
  delta,
  dim,
  degree = 3,
  type = "ps",
  quantile = FALSE,
  scale = TRUE,
  repara = FALSE,
  observed.region = FALSE,
  m2 = degree - 1,
  knot.margin = 0.001
) {
  # Create knot sequence for spline ----
  nk <- dim - degree + 1 # Number of "interior" knots (internal + boundary)

  # Make sure knot placement is within observed region
  xl <- min(t)
  xu <- max(t)
  xr <- xu - xl
  xl <- xl - xr * knot.margin # Left boundary knot
  xu <- xu + xr * knot.margin # Right boundary knot
  dx <- (xu - xl) / (nk - 1)
  knots <- seq(from = xl, to = xu, length.out = nk) # All knots
  internal_knots <- knots[2:(nk - 1)]
  boundary.knots <- c(xl, xu) # Boundary knots
  # knots <- seq(xl - dx * degree, xu + dx * degree, length = nk + 2 * degree) # Vector of knots
  # boundary.knots <- knots[c(1, length(knots))]
  # internal_knots <- knots[(degree + 2):(length(knots) - (degree + 1))]

  if (quantile) {
    # TODO Confirm this implementation is correct. Should be fine.
    if (observed.region) {
      internal_knots <- quantile(
        t[delta == 1],
        probs = seq(0, 1, length.out = nk)
      )[-c(1, nk)]
    } else {
      internal_knots <- quantile(
        t,
        probs = seq(0, 1, length.out = nk)
      )[-c(1, nk)]
    }
    knots[(degree + 2):(length(knots) - (degree + 1))] <- internal_knots
  }
  X <- splines2::naturalSpline(
    t,
    degree = degree,
    intercept = TRUE,
    knots = internal_knots,
    Boundary.knots = boundary.knots
  )
  # X <- splines::splineDesign(knots, t, degree + 1)

  # Create penalty matrix S = t(D1) %*% D1 if necessary ----
  if (type == "bs") {
    ## Integrated squared derivative penalty ----

    pord <- degree - m2
    k0 <- knots[(degree + 1):(degree + nk)]
    h <- diff(k0)
    h1 <- rep(h / pord, each = pord)
    k1 <- cumsum(c(k0[1], h1))

    # D <- splines::splineDesign(knots, k1, derivs = m2)
    D <- splines2::naturalSpline(
      k1,
      degree = degree,
      intercept = TRUE,
      knots = internal_knots,
      Boundary.knots = boundary.knots,
      derivs = m2
    )
    P <- solve(matrix(
      rep(seq(-1, 1, length.out = pord + 1), pord + 1)^rep(
        0:pord,
        each = pord + 1
      ),
      pord + 1,
      pord + 1
    ))
    i1 <- rep(1:(pord + 1), pord + 1) + rep(1:(pord + 1), each = pord + 1) ## i + j
    H <- matrix((1 + (-1)^(i1 - 2)) / (i1 - 1), pord + 1, pord + 1)
    W1 <- t(P) %*% H %*% P
    h <- h / 2 ## map integration interval to [-1,1] for maximum stability
    ## Create the non-zero diagonals of the W matrix...
    D1 <- compute_D1_cpp(D, W1, h, pord)
    # D1 <- compute_D1_R(D, W1, h, pord)
    S <- crossprod(D1)
  } else if (type == "ps") {
    ## Discrete penalty ----
    D1 <- diff(diag(dim), differences = m2)
    S <- crossprod(D1)
  } else if (type == "gps") {
    M1 <- M2 <- c()
    M <- diff(diag(dim))

    # W1
    # M1 <- diff(knots[2:(length(knots) - 1)], lag = 3) #Alternative way to compute M1 and M2
    for (i in 1:(dim - 1)) {
      M1[i] <- knots[degree + 1 + i] - knots[i + 1]
    }
    # W2
    # M2 <- diff(knots[3:(length(knots) - 2)], lag = 2)
    for (i in 1:(dim - 2)) {
      M2[i] <- knots[degree + 1 + i] - knots[i + 2]
    }

    W1 <- diag(M1) / (degree + 1 - 1)
    W2 <- diag(M2) / (degree + 1 - 2)

    D1 <- solve(W2) %*% diff(diag(dim - 1)) %*% solve(W1) %*% diff(diag(dim))
    S <- crossprod(D1)
  }

  # Scaling the penalty matrix S ----
  if (scale) {
    # maXX <- norm(X, type = "I")^2
    # maS <- norm(S) / maXX
    maXX <- norm(crossprod(X), type = 'F')
    maS <- norm(S, type = "F") / maXX
    S <- S / maS
    D1 <- D1 / sqrt(maS)
  } else {
    maS <- NULL
  }

  # Reparametrization NOT SUPPORTED
  if (repara) {
    # G <- t(splines::splineDesign(knots, seq(min(t),max(t),length=dim), degree+1))
    # Gm <- solve(G)
    # X <- X %*% Gm
    # S <-  t(Gm) %*% S %*% Gm
    sv <- svd(splines::splineDesign(
      knots,
      seq(min(t), max(t), length.out = dim),
      degree + 1
    ))
    if (sv$d[dim] / sv$d[1] < .Machine$double.eps^.66) {
      warning("Reparametrization unstable. Original model matrix returned")
      XP <- NULL
    } else {
      XP <- sv$v %*% (t(sv$u) / sv$d)
      X <- X %*% XP
      S <- t(XP) %*% S %*% XP
      # S <- S/eigen(S,symmetric=TRUE,only.values=TRUE)$values[1]
    }
  } else {
    XP <- NULL
  }

  return(list(
    X = X, # Note that this is a splines2 object
    S = S,
    D = D1,
    S.scale = maS,
    XP = XP
  ))
}

compute_D1_R <- function(D, W1, h, pord) {
  ld0 <- rep(sdiag_cpp(W1), length(h)) * rep(h, each = pord + 1)
  i1 <- c(
    rep(1:pord, length(h)) + rep(0:(length(h) - 1) * (pord + 1), each = pord),
    length(ld0)
  )
  ld <- ld0[i1] # extract elements for leading diagonal
  i0 <- 1:(length(h) - 1) * pord + 1
  i2 <- 1:(length(h) - 1) * (pord + 1)
  ld[i0] <- ld[i0] + ld0[i2] # add on extra parts for overlap
  B <- matrix(0, pord + 1, length(ld))
  B[1, ] <- ld
  for (k in 1:pord) {
    # create the other diagonals...
    diwk <- sdiag_cpp(W1, k) ## kth diagonal of W1
    ind <- 1:(length(ld) - k)
    B[k + 1, ind] <- (rep(h, each = pord) *
      rep(c(diwk, rep(0, k - 1)), length(h)))[ind]
  }
  # B now contains the non-zero diagonals of W
  B <- band_chol_cpp(B) # Banded cholesky factor.
  # Multiply D by the Cholesky factor
  D1 <- B[1, ] * D
  for (k in 1:pord) {
    ind <- 1:(nrow(D) - k)
    D1[ind, ] <- D1[ind, ] + B[k + 1, ind] * D[ind + k, ]
  }
}
