# toy simulation: outcome-aligned signal and AUROC
# required: ggplot2, pROC, randomForest, e1071, gbm, nnet

source("https://raw.githubusercontent.com/seungchan-an/utils/refs/heads/master/utils.R")

packages <- c("ggplot2", "pROC", "randomForest", "e1071", "gbm", "nnet")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("install required packages: ", paste(missing, collapse = ", "))

set.seed(123)
output_dir <- "data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

sample_sizes <- c(64, 128, 256, 512, 1024)
effect_sizes <- c(0, 0.5, 1, 2)
n_repeats <- 100L
n_features <- 8L
n_signal <- 2L
model_names <- c("LR", "RF", "SVM", "GBM", "MLP")

# fixed positive-class orientation; no automatic ROC direction selection
get_auc <- function(y, score) {
  as.numeric(pROC::roc(
    response = y, predictor = score,
    levels = c("0", "1"), direction = "<", quiet = TRUE
  )$auc)
}

simulate_once <- function(n, mu) {
  print(c(n, mu))
  # equally sized classes, 2 shifted features, 6 null features
  y <- factor(rep(c("0", "1"), each = n / 2), levels = c("0", "1"))
  x <- matrix(stats::rnorm(n * n_features), nrow = n, ncol = n_features)
  x[y == "1", seq_len(n_signal)] <- x[y == "1", seq_len(n_signal)] + mu
  colnames(x) <- paste0("x", seq_len(n_features))
  dat <- data.frame(y = y, x, check.names = FALSE)
  
  # stratified random 80/20 split; both classes represented in test set
  train_idx <- unlist(lapply(split(seq_len(n), y), function(ix) {
    sample(ix, size = floor(0.8 * length(ix)))
  }), use.names = FALSE)
  train <- dat[train_idx, , drop = FALSE]
  test <- dat[-train_idx, , drop = FALSE]
  x_test <- test[, paste0("x", seq_len(n_features)), drop = FALSE]
  
  lr <- stats::glm(y ~ ., data = train, family = stats::binomial())
  p_lr <- stats::predict(lr, newdata = test, type = "response")
  
  rf <- randomForest::randomForest(y ~ ., data = train)
  p_rf <- predict(rf, newdata = test, type = "prob")[, "1"]
  
  svm <- e1071::svm(y ~ ., data = train, probability = TRUE)
  svm_pred <- predict(svm, newdata = test, probability = TRUE)
  p_svm <- attr(svm_pred, "probabilities")[, "1"]
  
  gbm_train <- transform(train, y_num = as.integer(y == "1"))
  gbm_fit <- gbm::gbm(
    y_num ~ . - y, data = gbm_train, distribution = "bernoulli",
    n.trees = 150, verbose = FALSE
  )
  p_gbm <- predict(gbm_fit, newdata = test, n.trees = 150, type = "response")
  
  mlp <- nnet::nnet(y ~ ., data = train, size = 3, maxit = 300, trace = FALSE)
  p_mlp <- as.numeric(predict(mlp, newdata = x_test, type = "raw"))
  
  scores <- c(LR = get_auc(test$y, p_lr),
              RF = get_auc(test$y, p_rf),
              SVM = get_auc(test$y, p_svm),
              GBM = get_auc(test$y, p_gbm),
              MLP = get_auc(test$y, p_mlp))
  if (any(!is.finite(scores))) stop("non-finite AUROC")
  scores
}

# all simulated repetitions are retained for independent inspection
raw_rows <- vector("list", length(effect_sizes) * length(sample_sizes) * n_repeats)
i <- 0L
for (mu in effect_sizes) {
  for (n in sample_sizes) {
    for (rep_id in seq_len(n_repeats)) {
      scores <- simulate_once(n, mu)
      i <- i + 1L
      raw_rows[[i]] <- data.frame(
        effect_size = mu, n = n, repetition = rep_id,
        model = names(scores), AUROC = unname(scores)
      )
    }
  }
}
raw <- do.call(rbind, raw_rows)
utils::write.csv(raw, file.path(output_dir, "auroc_repetitions.csv"), row.names = FALSE)

# 95% intervals quantify Monte Carlo uncertainty of the mean AUROC
aggregate_group <- function(d) {
  m <- mean(d$AUROC)
  s <- stats::sd(d$AUROC)
  se <- s / sqrt(nrow(d))
  h <- stats::qt(0.975, df = nrow(d) - 1) * se
  data.frame(mean_AUROC = m, sd_AUROC = s, lower_95CI = m - h,
             upper_95CI = m + h, repetitions = nrow(d))
}
groups <- split(raw, interaction(raw$effect_size, raw$n, raw$model, drop = TRUE))
summary_rows <- lapply(groups, function(d) {
  cbind(d[1, c("effect_size", "n", "model")], aggregate_group(d))
})
summary <- do.call(rbind, summary_rows)
rownames(summary) <- NULL
summary <- summary[order(summary$effect_size, summary$n,
                         match(summary$model, model_names)), ]
utils::write.csv(summary, file.path(output_dir, "auroc_summary.csv"), row.names = FALSE)

summary$model <- factor(summary$model, levels = model_names)
summary$mu_label <- factor(
  paste0("mu = ", summary$effect_size),
  levels = paste0("mu = ", effect_sizes)
)

plot_auc <- ggplot2::ggplot(summary, ggplot2::aes(x = factor(n), y = model,
                                                  fill = mean_AUROC)) +
  ggplot2::geom_tile(color = "white", linewidth = line_width) +
  ggplot2::facet_wrap(~mu_label, nrow = 1) +
  scico::scale_fill_scico(
    palette = "batlow",
    limits = c(0.4, 1),
    breaks = c(0.4, 0.6, 0.8, 1),
    name = "AUC-ROC"
  ) +
  ggplot2::labs(x = "Total sample size (n)", y = NULL) +
  ggplot2::coord_equal() +
  theme_pub() +
  ggplot2::theme(panel.spacing = grid::unit(0.15, "cm"),
                 strip.background = ggplot2::element_blank(),
                 legend.position = "top",
                 axis.text.x = ggplot2::element_text(
                   size = 7, angle = 45, hjust = 1
                 ))

ggplot2::ggsave(file.path(output_dir, "figure_1d_auroc.pdf"), plot_auc,
                width = 7.5, height = 4.8, units = "cm")

# model settings, versions and exact runtime environment
capture.output(sessionInfo(), file = "session_info.txt")
cat("saved results in:", normalizePath(output_dir), "\n")

