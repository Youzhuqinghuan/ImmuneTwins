args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: Rscript baseline/perception.R expression.tsv counts.tsv genes.RDS models.RDS output.tsv")
}

suppressPackageStartupMessages(library(caret))

expression <- read.delim(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
rownames(expression) <- make.names(expression$Gene, unique = TRUE)
expression$Gene <- NULL
genes <- readRDS(args[[3]])
models <- readRDS(args[[4]])
required <- unique(unlist(lapply(models, function(entry) entry$model$coefnames)))
missing_required <- setdiff(required, rownames(expression))
if (length(missing_required) > 0) {
  stop(sprintf("expression is missing %d features used by the fitted models", length(missing_required)))
}
rank_normalized <- apply(
  expression[genes, , drop = FALSE],
  2,
  function(x) rank(x, ties.method = "average") / length(x)
)

viability <- sapply(
  models,
  function(entry) predict(entry$model, data.frame(t(rank_normalized)))
)
rownames(viability) <- colnames(rank_normalized)
viability_z <- apply(viability, 2, scale)
rownames(viability_z) <- rownames(viability)
combination <- pmin(viability_z[, "carfilzomib"], viability_z[, "lenalidomide"])

clone_names <- rownames(viability_z)
patient <- sub("_.*$", "", clone_names)
patient <- sub("^z\\.", "", patient)
clone <- sub("^[^_]+_", "", clone_names)
clone_scores <- data.frame(
  patient = patient,
  clone = clone,
  combination_viability_z = as.numeric(combination),
  stringsAsFactors = FALSE
)

counts <- read.delim(args[[2]], check.names = FALSE, stringsAsFactors = FALSE)
colnames(counts) <- c("patient", "c1", "c2", "c3")
score_patient <- function(patient_id) {
  available <- clone_scores[clone_scores$patient == patient_id, , drop = FALSE]
  if (nrow(available) == 0) return(NA_real_)
  row <- counts[counts$patient == patient_id, , drop = FALSE]
  if (nrow(row) != 1) stop(sprintf("missing or duplicated clone-count row for %s", patient_id))
  weights <- as.numeric(row[1, available$clone, drop = TRUE])
  weights <- weights / sum(weights)
  max(available$combination_viability_z * weights)
}

result <- data.frame(
  patient_id = counts$patient,
  perception_score = vapply(counts$patient, score_patient, numeric(1)),
  stringsAsFactors = FALSE
)
write.table(result, args[[5]], sep = "\t", quote = FALSE, row.names = FALSE)
