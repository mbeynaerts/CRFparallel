test_that("spline control provides knot margin for spline construction", {
  expect_identical(spline.control()$knot.margin, 0.001)
})
