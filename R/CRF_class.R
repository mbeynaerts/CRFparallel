class_coefficients <- S7::new_class(
  "class_coefficients",
  properties = list(
    coefficients = class_vector,
    label = new_property(class_vector, getter = function(self) {
      paste0("beta", 1:lenght(self@coefficients))
    })
  ),
  constructor = function(coef) {
    new_object(
      S7_object(),
      coefficients = coef
    )
  }
)

class_model.matrix <- S7::new_class(
  "class_model.matrix",
  properties = list(
    model.matrix = class_numeric,
    idx = class_vector
  ),
  constructor = function(model.matrix, idx) {
    new_object(
      S7_object(),
      model.matrix = model.matrix,
      idx = idx
    )
  }
)

#' @export
CRFspline <- S7::new_class(
  "CRFspline",
  properties = list(
    model.matrix = class_model.matrix,
    splines2 = new_property(
      class_list,
      validator = function(value) {
        if (length(value) != 2) {
          "@splines2 must be a list containing two splines2 objects"
        }
        if (
          !inherits(value[[1]], "splines2") ||
            !inherits(value[[1]], "splines2")
        ) {
          "At least one element of @splines2 is not a splines2 object."
        }
      },
      setter = function(self, value) {
        if (length(self@splines2) != 0) {
          cli_abort("@splines2 is read-only.")
        }
        self@splines2 <- value
        self
      }
    ),
    vcov = class_numeric,
    coefficients = class_coefficients,
    lambda = class_vector,
    fitted.values = new_property(class_vector, getter = function(self) {
      as.vector(
        self@model.matrix@model.matrix %*%
          self@coefficients@coefficients
      )
    }),
    loglik = class_double,
    reml = class_double,
    iterations = class_integer,
    call = class_call
  ),
  constructor = function(
    model.matrix,
    idx,
    splines2_list,
    vcov,
    coefficients,
    lambda,
    loglik,
    reml,
    iterations,
    call
  ) {
    new_object(
      S7_object(),
      model.matrix = class_model.matrix(model.matrix, idx),
      splines2 = splines2_list,
      vcov = vcov,
      coefficients = class_coefficients(coefficients),
      lambda = lambda,
      loglik = loglik,
      reml = reml,
      iterations = iterations,
      call = call
    )
  }
)

#' @export
CRFpoly <- S7::new_class(
  "CRFpoly",
  properties = list(
    degree = class_integer,
    restrict_degree = class_integer,
    model.matrix = class_model.matrix,
    vcov = class_numeric,
    coefficients = class_coefficients,
    fitted.values = new_property(class_vector, getter = function(self) {
      beta <- rep(0, (self@degree + 1)^2)
      beta[self@model.matrix@idx] <- self@coefficients@coefficients
      self@model.matrix@model.matrix %*% beta
    }),
    loglik = class_double,
    call = class_call
  ),
  constructor = function(
    method.args,
    model.matrix,
    idx,
    vcov,
    coefficients,
    loglik,
    call = call
  ) {
    new_object(
      S7_object(),
      degree = method.args$degree,
      restrict_degree = method.args$restrict_degree,
      model.matrix = class_model.matrix(model.matrix, idx),
      vcov = vcov,
      coefficients = class_coefficients(coefficients),
      loglik = loglik,
      call = call
    )
  }
)

#' @export
CRFbern <- S7::new_class(
  "CRFbern",
  properties = list(
    m = class_integer,
    tau = class_vector,
    marginal_km = new_property(class_list, validator = function(value) {
      if (length(value) != 3L) {
        "@marginal_km must be a list with exactly 3 survfit objects."
      }
      if (length(unique(sapply(value, class))) > 1L) {
        "@marginal_km must only contain survfit objects."
      }
    }),
    fitted.values = new_property(class_vector, getter = function(self) {
      as.vector(bernstein_estimator_vec(
        s1 = self@marginal_km$km1$surv,
        s2 = self@marginal_km$km2$surv,
        m = self@m,
        tau1 = self@tau[1],
        tau2 = self@tau[2]
      ))
    }),
    call = class_call
  ),
  constructor = function(method.args, tau, marginal_km, call) {
    new_object(
      S7_object(),
      m = method.args$m,
      tau = tau,
      marginal_km = marginal_km,
      call = call
    )
  }
)

# CRFspline.object <- function(method.args) {
#   if (!is.list(method.args)) {
#     cli::cli_abort("{.arg method.args} must be a {.cls list}.")
#   }
#   object <- CRF.object()
#   object$method <- "spline"
#   class(object) <- c("CRFspline", "CRF", "list")
#   object$model.matrix <- matrix()
#   object$vcov <- matrix()
#   object$coefficients <- numeric(length = method.args$dim^2)
#   attr(object$coefficients, "names") <- paste0("beta", 1:method.args$dim^2)
#   attr(object$model.matrix, "idx") <- 1:(method.args$dim^2)
#   attr(object$method, "type") <- method.args$type
#   attr(object$method, "dim") <- method.args$dim
#   attr(object$method, "degree") <- method.args$degree
#   attr(object$method, "scale") <- method.args$scale
#   attr(object$method, "lambda") <- numeric(length = 2)
#   attr(object$method, "knots") <- vector("list", length = 2)
#   attr(object$method, "splines2object") <- method.args$observed.region
#   attr(object$method, "quantile") <- method.args$quantile
#   attr(object$method, "iterations") <- integer()
#   object$loglik <- numeric()
#   object$reml <- numeric()
#   object$model <- data.frame()
# }

# CRFpoly.object <- function(method.args) {
#   if (!is.list(method.args)) {
#     cli::cli_abort("{.arg method.args} must be a {.cls list}.")
#   }
#   object <- CRF.object()
#   object$method <- "polynomial"
#   object$model.matrix <- matrix()
#   object$vcov <- matrix()
#   l <- (method.args$restrict_degree + 1) *
#     (method.args$restrict_degree + 2) /
#     2
#   object$coefficients <- numeric(length = l)
#   attr(object$coefficients, "names") <- paste0("beta", 1:l)
#   attr(object$model.matrix, "idx") <- integer()
#   attr(object$method, "degree") <- method.args$degree
#   attr(object$method, "restrict_degree") <- method.args$restrict_degree
#   object$loglik <- numeric()
#   object$reml <- NA
#   object$model <- data.frame()
#   class(object) <- c("CRFpoly", "CRF", "list")
# }

# CRFbern.object <- function(method.args) {
#   stopifnot(is.list(method.args))
#   object <- CRF.object()
#   object$method <- "bernstein"
#   object$m <- method.args$m
#   object$tau <- numeric(length = 2)
#   class(object) <- c("CRFbern", "CRF", "list")
#   return(object)
# }
