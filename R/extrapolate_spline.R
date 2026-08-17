# extrapolate_spline <- function(object, t, index) {
#   m <- attr(object$method, "degree") + 1
#   knots <- attr(object$method, "knots")[[index]]
#   nk <- length(knots)
#   ll <- knots[m]
#   ul <- knots[nk - (m - 1)]

#   ind <- t <= ul & t >= ll # Data in rage
#   n <- length(t) # Number of observations in range

#   # If all in range, just use standard model matrix
#   if (sum(ind) == n) {
#     X <- splines::splineDesign(knots, t, ord = m)
#   } else {
#     # Else do some extrapolation
#     # Mapping matrix
#     D <- splines::splineDesign(
#       knots = knots,
#       x = c(ll, ll, ul, ul),
#       ord = m,
#       derivs = c(0, 1, 0, 1)
#     )
#     X <- matrix(0, n, ncol(D))
#     if (sum(ind) > 0) {
#       X[ind, ] <- splines::splineDesign(knots, t[ind], ord = m)
#     } # interior rows
#     ind <- t < ll # First values smaller than lower bound
#     if (sum(ind) > 0) {
#       X[ind, ] <- cbind(1, t[ind] - ll) %*% D[1:2, ]
#     }
#     ind <- t > ul # Now values larger than upper bound
#     if (sum(ind) > 0) X[ind, ] <- cbind(1, t[ind] - ul) %*% D[3:4, ]
#   }

#   return(X)
# }

# extrapolate_spline <- function(object, t, index) {
#   m <- attr(object$method, "degree")
#   knots <- attr(object$method, "knots")[[index]]
#   internal.knots <- knots$internal.knots
#   boundary.knots <- knots$boundary.knots

#   X <- splines2::naturalSpline(
#     t,
#     knots = internal.knots,
#     degree = m,
#     intercept = TRUE,
#     Boundary.knots = boundary.knots
#   )

#   return(X)
# }
