test_that("spline control provides knot margin for spline construction", {
  expect_identical(CRFparallel::spline.control()$knot.margin, 0.001)
})

test_that("spline control provides initial smoothing parameters", {
  expect_identical(CRFparallel::spline.control()$lambda.init, c(1, 1))
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
