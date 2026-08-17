# save_fit_work <- function(fit, filepath, format) {
#   # First check whether filepath already contains extension
#   ext <- tools::file_ext(filepath)
#   if (ext == "") {
#     # No extension in filepath
#     # Create extension based on format
#     new_filepath <- paste0(filepath, ".", format)
#   } else {
#     # Extension in filepath
#     # Check whether ext = format
#     if (ext %in% c("rds", "qs2", "qdata", "qd")) {
#       new_filepath <- filepath
#       format <- ext
#     } else {
#       cli::cli_abort(
#         "The file extension in `file_path` is not supported. Please remove the extension or use a supported extension (see 'format')."
#       )
#     }
#   }

#   if (format == "rds") {
#     saveRDS(fit, file = new_filepath)
#   } else if (format == "qs2") {
#     qs2::qs_save(fit, file = new_filepath)
#   } else {
#     qs2::qs_save(fit, file = new_filepath)
#   }

#   invisible(new_filepath)
# }

# load_fit_work <- function(filepath) {
#   # Check filepath extension
#   ext <- tools::file_ext(filepath)
#   if (ext == "rds") {
#     fit <- readRDS(filepath)
#   } else if (ext == "qs2") {
#     fit <- qs2::qs_read(filepath)
#   } else if (ext %in% c("qdata", "qd")) {
#     fit <- qs2::qd_read(filepath)
#   } else {
#     fit <- cli::cli_abort("File type not supported.")
#   }

#   # Check whether fit is a CRF object
#   stopifnot(any(
#     S7_inherits("CRFspline"),
#     S7_inherits("CRFpoly"),
#     S7_inherits("CRFbern")
#   ))

#   return(fit)
# }

# save_fit <- new_generic(
#   "save_fit",
#   function(fit, filepath, format = c("rds", "qs2", "qdata")) {
#     S7_dispatch()
#   }
# )

# method(save_fit, CRFspline) <- function(
#   fit,
#   filepath,
#   format = c("rds", "qs2", "qdata", "qd")
# ) {
#   format <- match.call(format)
#   save_fit_work(fit, filepath, format = format)
# }

# method(save_fit, CRFpoly) <- function(
#   fit,
#   filepath,
#   format = c("rds", "qs2", "qdata", "qd")
# ) {
#   format <- match.call(format)
#   save_fit_work(fit, filepath, format = format)
# }

# method(save_fit, CRFbern) <- function(
#   fit,
#   filepath,
#   format = c("rds", "qs2", "qdata", "qd")
# ) {
#   format <- match.call(format)
#   save_fit_work(fit, filepath, format = format)
# }

# load_fit <- new_generic(
#   "load_fit",
#   function(filepath) {
#     S7_dispatch()
#   }
# )

# method(load_fit, CRFspline) <- function(filepath) load_fit_work(filpath)
# method(load_fit, CRFpoly) <- function(filepath) load_fit_work(filpath)
# method(load_fit, CRFbern) <- function(filepath) load_fit_work(filpath)
