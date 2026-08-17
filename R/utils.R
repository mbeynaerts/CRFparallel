tensor_product <- function(X1, X2, coef.vector) {
  coef.matrix <- matrix(
    coef.vector,
    nrow = ncol(X1),
    ncol = ncol(X2),
    byrow = FALSE
  )

  spline <- X1 %*% coef.matrix %*% t(X2)

  return(spline)
}

keep_indices <- function(poly_degree = 3, restrict_degree = 3) {
  dim_mat <- poly_degree + 1

  # Create a dummy matrix to find the indices
  # We use column-major order to match Armadillo/R defaults
  dummy_mat <- matrix(0, nrow = dim_mat, ncol = dim_mat)
  keep_indices <- c()

  k <- 0
  for (j in 1:dim_mat) {
    for (i in 1:dim_mat) {
      # Adjust for R's 1-based indexing: (i-1) + (j-1) <= poly_order
      if ((i - 1) + (j - 1) <= restrict_degree) {
        # This is the linear index (0-based for C++)
        keep_indices <- c(keep_indices, k)
      }
      k <- k + 1
    }
  }

  # Output: indices of coefficients to keep in the (poly_degree+1) x (poly_degree+1) coefficient matrix
  # These indices are C++ indices, for R use keep_indices+1
  return(as.integer(keep_indices))
}

check_knots <- function(knots, type, dim, degree) {
  # Check whether exactly two list elements are provided
  if (length(knots) != 2) {
    cli::cli_abort(
      "The provided list of knots must only contain two elements corresponding to the knot vectors for each dimension."
    )
  }
  knots1 <- knots[[1]]
  knots2 <- knots[[2]]
  # Check whether the amount of supplied knots is in agreement with dim and degree
  nk <- dim + degree - 1
  if (all(sapply(knots, \(x) length(x) == nk))) {
    # Check whether knots are ordered
    if (!is.unsorted(knots1) | !is.unsorted(knots2)) {
      cli::cli_alert("Knots should be ordered in ascending sequence.")
    }
    if (type == "ps") {
      # Check whether knots for P-splines are evenly spaced
      if (!all.equal(diff(knots1)) || !all.equal(diff(knots2))) {
        cli::cli_abort(
          "Knots should be evenly spaced when specifying type 'ps'."
        )
      }
    }
  } else {
    cli::cli_abort("At least one knot vector does not have the correct length.")
  }
  return(invisible(knots))
}
