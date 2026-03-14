# =========================================================
# LIBRARIES
# =========================================================
library(tidyverse)
library(lubridate)
library(fastDummies)
library(plotrix)
library(corrplot)
library(pROC)
library(glmnet)
library(doParallel)
library(rpart)
library(randomForest)
library(smotefamily)
library(xgboost)
library(tidymodels)
library(finetune)
library(caret)

# =========================================================
# LOADING DATA
# =========================================================
dat <- read.csv("returns_train.csv", stringsAsFactors = FALSE)

# Make target numeric 0/1 just in case
dat$returned <- as.integer(dat$returned)

# =========================================================
# TRAIN / TEST SPLIT (70/30, stratified on returned)
# =========================================================
set.seed(42)
train_idx <- caret::createDataPartition(dat$returned, p = 0.7, list = FALSE)

datTrain <- dat[train_idx, , drop = FALSE]
datTest  <- dat[-train_idx, , drop = FALSE]

# =========================================================
# CLEANING FUNCTION
# =========================================================
data_cleaning <- function(df) {
  
  # --- customer_age ---
  df$customer_age[df$customer_age < 13] <- NA
  df$customer_age_missing <- as.integer(is.na(df$customer_age))
  df$customer_age[is.na(df$customer_age)] <- 0
  
  # --- account_age_days ---
  df$account_age_missing <- as.integer(is.na(df$account_age_days))
  df$account_age_days[is.na(df$account_age_days)] <- 0
  
  # --- payment_method ---
  df$payment_method <- tolower(trimws(as.character(df$payment_method)))
  df$payment_method[df$payment_method == "credit card - visa"] <- "visa"
  df$payment_method <- factor(df$payment_method)
  
  # --- guest_checkout ---
  guest_raw <- trimws(tolower(as.character(df$guest_checkout)))
  df$guest_checkout <- as.integer(guest_raw %in% c("true", "1", "t", "yes", "y"))
  
  # --- remove IDs if present ---
  if ("transaction_id" %in% names(df)) df$transaction_id <- NULL
  if ("customer_id" %in% names(df)) df$customer_id <- NULL
  
  # --- loyalty tier -> ordinal ---
  if ("loyalty_tier" %in% names(df)) {
    df$loyalty_tier_ordinal <- as.integer(factor(
      as.character(df$loyalty_tier),
      levels = c("None", "Bronze", "Silver", "Gold", "Platinum")
    )) - 1
    df$loyalty_tier <- NULL
  }
  
  # --- target ---
  df$returned <- as.integer(df$returned)
  
  return(df)
}

# =========================================================
# FEATURE ENGINEERING FUNCTION
# =========================================================
data_features <- function(df) {
  
  # --- timestamp ---
  df$transaction_timestamp <- lubridate::ymd_hms(as.character(df$transaction_timestamp), quiet = TRUE)
  
  df$hour_of_day <- lubridate::hour(df$transaction_timestamp)
  df$day_of_week <- lubridate::wday(df$transaction_timestamp)
  df$month <- lubridate::month(df$transaction_timestamp)
  df$is_weekend <- as.integer(df$day_of_week %in% c(1, 7))
  
  # cyclical encoding
  df$hour_sin <- sin(2 * pi * df$hour_of_day / 24)
  df$hour_cos <- cos(2 * pi * df$hour_of_day / 24)
  df$day_sin  <- sin(2 * pi * df$day_of_week / 7)
  df$day_cos  <- cos(2 * pi * df$day_of_week / 7)
  
  # holiday-ish season flag
  df$is_holiday_season <- as.integer(df$month %in% c(11, 12))
  
  df$transaction_timestamp <- NULL
  
  # --- zip feature ---
  zip_first_digit <- suppressWarnings(as.integer(substr(sprintf("%05s", df$zip_code), 1, 1)))
  df$zip_region <- cut(
    zip_first_digit,
    breaks = c(-Inf, 2, 4, 6, 7, Inf),
    labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
    right = TRUE
  )
  df$zip_code <- NULL
  
  # --- promo features ---
  promo_text <- as.character(df$applied_promo_codes)
  df$promo_FREESHIP <- as.integer(grepl("FREESHIP", promo_text))
  df$promo_NEWUSER  <- as.integer(grepl("NEWUSER", promo_text))
  df$promo_SAVE20   <- as.integer(grepl("SAVE20", promo_text))
  df$promo_WINTER50 <- as.integer(grepl("WINTER50", promo_text))
  df$num_promos <- df$promo_FREESHIP + df$promo_NEWUSER + df$promo_SAVE20 + df$promo_WINTER50
  df$applied_promo_codes <- NULL
  
  # --- binned age features ---
  df$customer_age_bin <- cut(
    df$customer_age,
    breaks = c(-Inf, 18, 30, 45, 60, Inf),
    labels = c("Under18", "18to30", "31to45", "46to60", "Over60"),
    right = FALSE
  )
  
  df$account_age_bin <- cut(
    df$account_age_days,
    breaks = c(-Inf, 104, 250, 504, Inf),
    labels = c("New", "Developing", "Established", "Loyal"),
    right = FALSE
  )
  
  # --- numeric transforms ---
  df$log_price <- log1p(df$price)
  df$log_account_age <- log1p(df$account_age_days)
  df$discount_value <- df$price * df$discount_pct
  df$final_price <- df$price * (1 - df$discount_pct)
  df$discount_strength <- df$discount_value / (df$final_price + 1)
  
  # --- interactions ---
  df$guest_discount <- df$guest_checkout * df$discount_pct
  df$price_loyalty <- df$price * df$loyalty_tier_ordinal
  df$price_x_freeship <- df$price * df$promo_FREESHIP
  df$discount_x_newuser <- df$discount_pct * df$promo_NEWUSER
  df$guest_x_newuser <- df$guest_checkout * df$promo_NEWUSER
  
  # convert key character cols to factor before dummying
  factor_cols <- intersect(
    c("marketing_channel", "product_category", "product_subcategory",
      "payment_method", "customer_age_bin", "account_age_bin", "zip_region"),
    names(df)
  )
  df[factor_cols] <- lapply(df[factor_cols], factor)
  
  return(df)
}

# =========================================================
# TRAIN-ONLY FEATURE MAPS (NO TARGET LEAKAGE)
# =========================================================
fit_feature_maps <- function(train_df) {
  
  # quantile cutoffs from TRAIN only
  price_q75 <- as.numeric(stats::quantile(train_df$price, 0.75, na.rm = TRUE))
  discount_q75 <- as.numeric(stats::quantile(train_df$discount_pct, 0.75, na.rm = TRUE))
  
  # category mean price from TRAIN only
  category_price_map <- train_df %>%
    group_by(product_category) %>%
    summarise(category_avg_price = mean(price, na.rm = TRUE), .groups = "drop")
  
  # subcategory mean price from TRAIN only
  subcategory_price_map <- train_df %>%
    group_by(product_subcategory) %>%
    summarise(subcategory_avg_price = mean(price, na.rm = TRUE), .groups = "drop")
  
  list(
    price_q75 = price_q75,
    discount_q75 = discount_q75,
    category_price_map = category_price_map,
    subcategory_price_map = subcategory_price_map
  )
}

apply_feature_maps <- function(df, maps) {
  
  overall_avg_price <- mean(df$price, na.rm = TRUE)
  
  # category price map
  df <- df %>%
    left_join(maps$category_price_map, by = "product_category")
  df$category_avg_price[is.na(df$category_avg_price)] <- overall_avg_price
  df$price_vs_category <- df$price / (df$category_avg_price + 1e-8)
  
  # subcategory price map
  df <- df %>%
    left_join(maps$subcategory_price_map, by = "product_subcategory")
  df$subcategory_avg_price[is.na(df$subcategory_avg_price)] <- overall_avg_price
  df$price_vs_subcategory <- df$price / (df$subcategory_avg_price + 1e-8)
  
  # threshold features
  df$high_price <- as.integer(df$price > maps$price_q75)
  df$high_discount <- as.integer(df$discount_pct > maps$discount_q75)
  df$expensive_discounted <- df$high_price * df$high_discount
  
  # remove helper columns
  df$category_avg_price <- NULL
  df$subcategory_avg_price <- NULL
  
  return(df)
}

# =========================================================
# DUMMY CREATION + ALIGNMENT
# =========================================================
make_dummies_aligned <- function(train_df, test_df) {
  
  dummy_vars <- c(
    "marketing_channel",
    "product_category",
    "product_subcategory",
    "payment_method",
    "customer_age_bin",
    "account_age_bin",
    "zip_region"
  )
  
  dummy_vars <- dummy_vars[dummy_vars %in% names(train_df)]
  
  train_df <- fastDummies::dummy_cols(
    train_df,
    select_columns = dummy_vars,
    remove_first_dummy = TRUE,
    remove_selected_columns = TRUE
  )
  
  test_df <- fastDummies::dummy_cols(
    test_df,
    select_columns = dummy_vars,
    remove_first_dummy = TRUE,
    remove_selected_columns = TRUE
  )
  
  # drop redundant NA dummy bins if they happen to exist
  cols_to_drop <- c("customer_age_bin_NA", "account_age_bin_NA")
  train_df <- train_df[, !(names(train_df) %in% cols_to_drop), drop = FALSE]
  test_df  <- test_df[, !(names(test_df) %in% cols_to_drop), drop = FALSE]
  
  # add missing columns to test
  missing_in_test <- setdiff(names(train_df), names(test_df))
  for (col in missing_in_test) {
    test_df[[col]] <- 0
  }
  
  # remove extra test columns
  extra_in_test <- setdiff(names(test_df), names(train_df))
  if (length(extra_in_test) > 0) {
    test_df <- test_df[, !(names(test_df) %in% extra_in_test), drop = FALSE]
  }
  
  # reorder test columns to match train
  test_df <- test_df[, names(train_df), drop = FALSE]
  
  return(list(train = train_df, test = test_df))
}

# =========================================================
# PREPROCESS TRAIN AND TEST
# =========================================================
datTrain <- data_cleaning(datTrain)
datTest  <- data_cleaning(datTest)

feature_maps <- fit_feature_maps(
  data_features(datTrain)
)

datTrain_processed <- datTrain %>%
  data_features() %>%
  apply_feature_maps(feature_maps)

datTest_processed <- datTest %>%
  data_features() %>%
  apply_feature_maps(feature_maps)

dummy_out <- make_dummies_aligned(datTrain_processed, datTest_processed)
datTrain_processed <- dummy_out$train
datTest_processed  <- dummy_out$test

# Make sure everything except returned is numeric
datTrain_processed$returned <- as.integer(datTrain_processed$returned)
datTest_processed$returned  <- as.integer(datTest_processed$returned)

# Remove near-zero variance columns based on TRAIN only
nzv_cols <- caret::nearZeroVar(datTrain_processed %>% select(-returned), saveMetrics = TRUE)
nzv_names <- rownames(nzv_cols)[nzv_cols$nzv]

if (length(nzv_names) > 0) {
  datTrain_processed <- datTrain_processed[, !(names(datTrain_processed) %in% nzv_names), drop = FALSE]
  datTest_processed  <- datTest_processed[, !(names(datTest_processed) %in% nzv_names), drop = FALSE]
}

# Final alignment check
datTest_processed <- datTest_processed[, names(datTrain_processed), drop = FALSE]

cat("Train rows:", nrow(datTrain_processed), "\n")
cat("Test rows :", nrow(datTest_processed), "\n")
cat("Same columns in train/test:", identical(names(datTrain_processed), names(datTest_processed)), "\n\n")

# =========================================================
# OPTIONAL SMOTE ON TRAIN ONLY
# Applied only if minority class proportion < 25%
# =========================================================
y_train_original <- datTrain_processed$returned
class_props <- prop.table(table(y_train_original))
minority_prop <- min(class_props)

cat("Training class proportions:\n")
print(class_props)
cat("\nMinority proportion:", round(minority_prop, 4), "\n\n")

if (minority_prop < 0.25) {
  cat("Applying SMOTE to TRAIN only...\n")
  
  smote_out <- smotefamily::SMOTE(
    X = datTrain_processed %>% select(-returned),
    target = as.factor(datTrain_processed$returned),
    K = 5,
    dup_size = 0
  )
  
  train_model_df <- smote_out$data
  names(train_model_df)[ncol(train_model_df)] <- "returned"
  train_model_df$returned <- as.integer(as.character(train_model_df$returned))
  
} else {
  cat("SMOTE not needed. Using original TRAIN data.\n")
  train_model_df <- datTrain_processed
}

# =========================================================
# MODEL MATRICES / DATA
# =========================================================
x_train <- as.matrix(train_model_df %>% select(-returned))
y_train <- train_model_df$returned

x_test <- as.matrix(datTest_processed %>% select(-returned))
y_test <- datTest_processed$returned

train_class_df <- train_model_df
train_class_df$returned_factor <- factor(train_class_df$returned, levels = c(0, 1), labels = c("No", "Yes"))
train_class_df$returned <- NULL

test_class_df <- datTest_processed
test_class_df$returned_factor <- factor(test_class_df$returned, levels = c(0, 1), labels = c("No", "Yes"))
test_class_df$returned <- NULL

# =========================================================
# AUC FUNCTION
# =========================================================
calc_auc <- function(actual, pred) {
  as.numeric(pROC::roc(response = actual, predictor = pred, quiet = TRUE)$auc)
}

# =========================================================
# 1) GLMNET (REGULARIZED LOGISTIC REGRESSION)
# =========================================================
set.seed(42)
glmnet_cv <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0.5,              # elastic net
  nfolds = 5,
  type.measure = "auc",
  standardize = TRUE
)

glmnet_pred <- as.numeric(
  predict(glmnet_cv, newx = x_test, s = "lambda.1se", type = "response")
)

glmnet_auc <- calc_auc(y_test, glmnet_pred)

cat("GLMNET AUC:", round(glmnet_auc, 4), "\n")

# =========================================================
# 2) DECISION TREE
# =========================================================
set.seed(42)
tree_model <- rpart(
  returned_factor ~ .,
  data = train_class_df,
  method = "class",
  control = rpart.control(
    cp = 0.002,
    maxdepth = 5,
    minsplit = 40,
    minbucket = 20,
    xval = 10
  )
)

tree_pred <- predict(tree_model, newdata = test_class_df, type = "prob")[, "Yes"]
tree_auc <- calc_auc(y_test, tree_pred)

cat("Decision Tree AUC:", round(tree_auc, 4), "\n")

# =========================================================
# RANDOM FOREST
# =========================================================

rf_train_df <- as.data.frame(x_train)
rf_test_df  <- as.data.frame(x_test)

colnames(rf_train_df) <- make.names(colnames(rf_train_df))
colnames(rf_test_df)  <- make.names(colnames(rf_test_df))

set.seed(42)

rf_model <- randomForest(
  x = rf_train_df,
  y = factor(y_train, levels = c(0, 1), labels = c("No", "Yes")),
  ntree = 500,
  mtry = max(2, floor(sqrt(ncol(rf_train_df)))),
  nodesize = 10,
  importance = TRUE
)

rf_pred <- predict(rf_model, newdata = rf_test_df, type = "prob")[, "Yes"]
rf_auc <- calc_auc(y_test, rf_pred)

cat("Random Forest AUC:", round(rf_auc, 4), "\n")

# =========================================================
# 4) XGBOOST (BEST DEFAULT CHOICE FOR AUC)
# =========================================================
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

# If SMOTE was NOT used and classes are imbalanced, use class weight
used_smote <- nrow(train_model_df) > nrow(datTrain_processed)
neg <- sum(y_train == 0)
pos <- sum(y_train == 1)
scale_pos_weight_value <- if (used_smote) 1 else neg / max(pos, 1)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.03,
  subsample = 0.8,
  colsample_bytree = 0.7,
  min_child_weight = 5,
  gamma = 1,
  lambda = 1,
  alpha = 0.5,
  scale_pos_weight = scale_pos_weight_value
)

set.seed(42)
xgb_cv <- xgb.cv(
  params = xgb_params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  stratified = TRUE,
  maximize = TRUE,
  early_stopping_rounds = 50,
  verbose = 0
)

best_nrounds <- which.max(xgb_cv$evaluation_log$test_auc_mean)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

xgb_pred <- predict(xgb_model, newdata = dtest)
xgb_auc <- calc_auc(y_test, xgb_pred)

cat("XGBoost AUC:", round(xgb_auc, 4), "\n")

# =========================================================
# RESULTS TABLE
# =========================================================
results <- tibble(
  Model = c("GLMNET", "Decision Tree", "Random Forest", "XGBoost"),
  AUC = c(glmnet_auc, tree_auc, rf_auc, xgb_auc)
) %>%
  arrange(desc(AUC))

cat("\n============================\n")
cat("MODEL COMPARISON (TEST AUC)\n")
cat("============================\n")
print(results)

# =========================================================
# BEST MODEL
# =========================================================
best_model_name <- results$Model[1]
cat("\nBest model on test AUC:", best_model_name, "\n")

# =========================================================
# FEATURE IMPORTANCE FOR XGBOOST
# =========================================================
xgb_importance <- xgb.importance(
  feature_names = colnames(x_train),
  model = xgb_model
)

print(head(xgb_importance, 20))
xgb.plot.importance(xgb_importance[1:20, ])

# =========================================================
# FEATURE IMPORTANCE FOR RANDOM FOREST
# =========================================================
rf_importance <- importance(rf_model)
print(rf_importance[order(rf_importance[, 1], decreasing = TRUE), , drop = FALSE][1:20, , drop = FALSE])

# =========================================================
# OPTIONAL: ROC CURVES FOR ALL MODELS
# =========================================================
roc_glmnet <- roc(y_test, glmnet_pred, quiet = TRUE)
roc_tree   <- roc(y_test, tree_pred, quiet = TRUE)
roc_rf     <- roc(y_test, rf_pred, quiet = TRUE)
roc_xgb    <- roc(y_test, xgb_pred, quiet = TRUE)

plot(roc_xgb, col = "red", main = "ROC Curves on Test Set")
plot(roc_rf, add = TRUE, col = "blue")
plot(roc_glmnet, add = TRUE, col = "darkgreen")
plot(roc_tree, add = TRUE, col = "orange")
legend(
  "bottomright",
  legend = c(
    paste0("XGBoost: ", round(xgb_auc, 3)),
    paste0("Random Forest: ", round(rf_auc, 3)),
    paste0("GLMNET: ", round(glmnet_auc, 3)),
    paste0("Decision Tree: ", round(tree_auc, 3))
  ),
  col = c("red", "blue", "darkgreen", "orange"),
  lwd = 2,
  bty = "n"
)


# =========================================================
# KAGGLE TEST PROCESSING + SUBMISSION FILE
# =========================================================


# Load unseen Kaggle test data
test <- read.csv("returns_test.csv", stringsAsFactors = FALSE)

# Save transaction_id for submission BEFORE cleaning
test_ids <- test$transaction_id

data_cleaning <- function(df) {
  
  # Customer age
  df$customer_age[df$customer_age < 13] <- NA
  df$customer_age_missing <- as.integer(is.na(df$customer_age))
  df$customer_age[is.na(df$customer_age)] <- 0
  
  # Account age
  df$account_age_missing <- as.integer(is.na(df$account_age_days))
  df$account_age_days[is.na(df$account_age_days)] <- 0
  
  # Payment method
  df$payment_method <- tolower(trimws(as.character(df$payment_method)))
  df$payment_method[df$payment_method == "credit card - visa"] <- "visa"
  df$payment_method <- factor(df$payment_method)
  
  # Guest checkout
  guest_raw <- trimws(tolower(as.character(df$guest_checkout)))
  df$guest_checkout <- as.integer(guest_raw %in% c("true", "1", "t", "yes", "y"))
  
  # Remove IDs if present
  if ("transaction_id" %in% names(df)) df$transaction_id <- NULL
  if ("customer_id" %in% names(df)) df$customer_id <- NULL
  
  # Loyalty tier
  if ("loyalty_tier" %in% names(df)) {
    df$loyalty_tier_ordinal <- as.integer(factor(
      as.character(df$loyalty_tier),
      levels = c("None", "Bronze", "Silver", "Gold", "Platinum")
    )) - 1
    df$loyalty_tier <- NULL
  }
  
  # Only convert returned if it exists
  if ("returned" %in% names(df)) {
    df$returned <- as.integer(df$returned)
  }
  
  return(df)
}
# Clean test data
test <- data_cleaning(test)

# Feature engineering
test <- data_features(test)

# Apply training-only feature maps learned from datTrain
test <- apply_feature_maps(test, feature_maps)

# Create dummies and align test columns to training columns
dummy_out <- make_dummies_aligned(
  train_df = datTrain_processed[, setdiff(names(datTrain_processed), "returned"), drop = FALSE],
  test_df  = test
)

test_processed <- dummy_out$test

# Make sure test has exactly the same predictor columns as training
train_cols <- setdiff(names(datTrain_processed), "returned")

for (col in setdiff(train_cols, names(test_processed))) {
  test_processed[[col]] <- 0
}

test_processed <- test_processed[, train_cols, drop = FALSE]

# Final NA cleanup just in case
for (col in names(test_processed)) {
  if (anyNA(test_processed[[col]])) {
    if (is.numeric(test_processed[[col]]) || is.integer(test_processed[[col]])) {
      test_processed[[col]][is.na(test_processed[[col]])] <- 0
    }
  }
}

# Predict with XGBoost
kaggle_pred <- predict(
  xgb_model,
  xgb.DMatrix(as.matrix(test_processed))
)

# Build submission file
submission <- data.frame(
  transaction_id = test_ids,
  returned = kaggle_pred
)

# Save submission
write.csv(submission, "submission_8.csv", row.names = FALSE)
cat("Submission saved: submission_8.csv\n")


# =========================================================
# KAGGLE TEST PROCESSING + SUBMISSION
# =========================================================

# Load unseen test data
test <- read.csv("returns_test.csv", stringsAsFactors = FALSE)

# Save IDs for submission
test_ids <- test$transaction_id

# Clean
test <- data_cleaning(test)

# Feature engineering
test <- data_features(test)

# Apply train-derived feature maps
test <- apply_feature_maps(test, feature_maps)

# Create dummies on test only
dummy_vars <- c(
  "marketing_channel",
  "product_category",
  "product_subcategory",
  "payment_method",
  "customer_age_bin",
  "account_age_bin",
  "zip_region"
)

dummy_vars <- dummy_vars[dummy_vars %in% names(test)]

test <- fastDummies::dummy_cols(
  test,
  select_columns = dummy_vars,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# Drop redundant NA dummy columns if present
cols_to_drop <- c("customer_age_bin_NA", "account_age_bin_NA")
test <- test[, !(names(test) %in% cols_to_drop), drop = FALSE]

# Remove returned if somehow present
if ("returned" %in% names(test)) {
  test$returned <- NULL
}

# Match training predictor columns exactly
train_cols <- setdiff(names(datTrain_processed), "returned")

# Add any missing columns to test
for (col in setdiff(train_cols, names(test))) {
  test[[col]] <- 0
}

# Remove any extra columns not used in training
extra_cols <- setdiff(names(test), train_cols)
if (length(extra_cols) > 0) {
  test <- test[, !(names(test) %in% extra_cols), drop = FALSE]
}

# Reorder columns exactly like training
test <- test[, train_cols, drop = FALSE]

# Final NA cleanup
for (col in names(test)) {
  if (anyNA(test[[col]])) {
    if (is.numeric(test[[col]]) || is.integer(test[[col]])) {
      test[[col]][is.na(test[[col]])] <- 0
    }
  }
}

# Predict with XGBoost
kaggle_pred <- predict(
  xgb_model,
  xgb.DMatrix(as.matrix(test))
)

# Submission
submission <- data.frame(
  transaction_id = test_ids,
  returned = kaggle_pred
)

write.csv(submission, "submission_9.csv", row.names = FALSE)
cat("Submission saved: submission_8.csv\n")
