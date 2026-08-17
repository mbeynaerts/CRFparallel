test_that("spline control provides knot margin for spline construction", {
  expect_identical(CRFparallel::spline.control()$knot.margin, 0.001)
})

test_that("spline control provides initial smoothing parameters", {
  expect_identical(CRFparallel::spline.control()$lambda.init, c(1, 1))
})

test_that("risk sets count observations above both margins", {
  x <- c(4, 1, 3, 2)
  y <- c(1, 4, 2, 3)

  expected <- outer(seq_along(x), seq_along(y), Vectorize(function(j, i) {
    sum(x >= x[j] & y >= y[i])
  }))

  expect_equal(CRFparallel:::riskset_fast(x, y), expected)
})

test_that("spline penalties match constructed natural spline basis", {
  x <- seq(1, 10, length.out = 20)
  obj <- CRFparallel:::construct_spline(
    t = x,
    delta = rep(1, length(x)),
    dim = 10,
    degree = 3,
    type = "ps"
  )

  expect_equal(dim(obj$S), c(ncol(obj$X), ncol(obj$X)))
})

test_that("tensor penalties match tensor coefficient count", {
  x <- seq(1, 10, length.out = 20)
  obj1 <- CRFparallel:::construct_spline(x, rep(1, length(x)), dim = 10)
  obj2 <- CRFparallel:::construct_spline(x, rep(1, length(x)), dim = 10)
  S <- CRFparallel:::construct_tensor_penalty(obj1, obj2)
  ncoef <- ncol(obj1$X) * ncol(obj2$X)

  expect_equal(dim(S$S1), c(ncoef, ncoef))
  expect_equal(dim(S$S2), c(ncoef, ncoef))
})
