model.frame <- function(formula, ...) {
  if (inherits(formula, "formula")) {
    return(model.frame.formula(formula, ...))
  }

  stats::model.frame(formula, ...)
}

#' @export
model.frame.formula <- function(formula, data, ...) {
  if (missing(data)) {
    data <- environment(formula)
  }

  terms <- terms(formula, data = data)
  if (attr(terms, "response") == 0L) {
    return(stats:::model.frame.default(formula = formula, data = data, ...))
  }

  response <- eval(attr(terms, "variables")[[2L]], data, environment(formula))

  if (!S7_inherits(response, biSurv)) {
    cli::cli_alert(glue::glue(
      "{formula} does not contain a {.cls biSurv} object."
    ))
    return(stats:::model.frame.default(formula = formula, data = data, ...))
  }

  out <- data.frame(response = I(list(response)))
  attr(out, "terms") <- terms
  out
}

model.response <- function(data, type = "any") {
  response <- data[[1L]]

  if (
    inherits(response, "AsIs") &&
      length(response) == 1L &&
      S7_inherits(response[[1L]], biSurv)
  ) {
    return(response[[1L]])
  }

  stats::model.response(data, type = type)
}
