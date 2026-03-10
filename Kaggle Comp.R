
# Libraries ---------------------------------------------------------------

library(tidyverse)
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


# Load Data ---------------------------------------------------------------

dat <- read.csv("returns_train.csv", stringsAsFactors = TRUE)
# Checking out the structure of the data 
dim(dat)
str(dat) # Returned = 1, kept = 0 

# EDA -------------------------------------------------

#Checking how many NAs we have. # NAs in account_age_days are for those people who have checked out as GUESTS 
colSums(is.na(dat)) 
# Customer age and account_age_days have NAs. The numbers are the same for both variables (10491)

# Checking if the NA's of account_age_days match the NA's of customer_age. If so, it's because they are checking out as GUESTS and that info is not available.  
dat[is.na(dat$account_age_days), "customer_age"] # All NAs in account_age_days are for customers with NA in customer_age. 
#We keep this NAs as they are because they are informative. 
#Missing at Random = the missingness is explained by other variables. 
      #in this case customer_age is missing because account_days_age is missing 
      #and checking out as guests does not allow for this info to be included 


# Visualizing class balance of the outcome variable 
barplot(table(dat$return), main = "Balance of the outcome variable", xlab = "Return", ylab = "Frequency")
# Checking the percentage of returns in the data 
sum(dat$return == 1) / nrow(dat) # 15 percent of the orders were returned (=1). Moderate degree of imbalance 
prop.table(table(dat$return)) # Gives me table with % of 0 and 1 
# Because there is moderate imbalance in the dataset, we can do undersampling or oversampling (SMOTE) to balance the classes 

# Checking for correlated variables
# Correlation between customer age and guest checkout. NAs in customer age are correlated 1 in guest checkouts. 
table(is.na(dat$customer_age), dat$guest_checkout, 
      dnn = c("customer_age is NA", "guest_checkout"))

table(is.na(dat$account_age_days), dat$guest_checkout, 
      dnn = c("customer_age is NA", "guest_checkout"))

# Correlation matrix on raw data - check feature redundancy
feature_matrix <- cor(dat[, sapply(dat, is.numeric) & names(dat) != "returned"],
                      use = "pairwise.complete.obs")
corrplot(feature_matrix, method = "color", tl.cex = 0.6)

# Correlation with target variable
target_cor <- cor(dat[, sapply(dat, is.numeric)],
                  use = "pairwise.complete.obs")[, "returned"]
sort(target_cor, decreasing = TRUE)

# Split the data ----------------------------------------------------------

set.seed(42)
train <- sample(1:nrow(dat), nrow(dat)*0.7)
datTrain <- dat[train, ]
datTest <- dat[-train, ]


# Data Cleaning  ----------------------------------------

data_cleaning <- function(df) {
  
  # Make negative ages into NAs
  df$customer_age[df$customer_age < 13] <- NA
  
  # Clean up payment method 
  df$payment_method <- factor(tolower(as.character(df$payment_method))) 
  df$payment_method <- recode(df$payment_method, "credit card - visa" = "visa")
  
  # Make guest_checkout a 0/1 variable
  df$guest_checkout <- as.integer(df$guest_checkout == "True")
  
  # Remove product subcategory - too complex
  #df$product_subcategory <- NULL
  
  # Remove unuseful variables
  df$transaction_id <- NULL
  df$customer_id <- NULL
  
  df$loyalty_tier_ordinal<- as.integer(factor(as.character(df$loyalty_tier), 
                                               levels = c("None", "Bronze", "Silver", "Gold", "Platinum")))
  df$loyalty_tier <- NULL
  
  return(df)
  
}

datTrain <- data_cleaning(datTrain)
datTest <- data_cleaning(datTest)


# Feature Engineering  ----------------------------------------------------

data_features <- function(df) {
  
  # Make timestamps a date-time object 
  df$transaction_timestamp <- ymd_hms(as.character(df$transaction_timestamp))
  
  # Make timestamps feature variables
  df$hour_of_day <- hour(df$transaction_timestamp)
  df$day_of_week <- wday(df$transaction_timestamp)
  df$month <- month(df$transaction_timestamp)
  df$is_weekend <- as.integer(wday(df$transaction_timestamp) %in% c(1, 7))
  df$hour_sin <- sin(2 * pi * df$hour_of_day / 24)
  df$hour_cos <- cos(2 * pi * df$hour_of_day / 24)
  df$day_sin  <- sin(2 * pi * df$day_of_week / 7)
  df$day_cos  <- cos(2 * pi * df$day_of_week / 7)
  #Remove unfeatured transaction timestamp
  df$transaction_timestamp <- NULL
  
  # Make zip code features
  df$zip_region <- cut(as.integer(substr(df$zip_code, 1, 1)),
                       breaks = c(-Inf, 2, 4, 6, 7, Inf),
                       labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                       right = TRUE)
  #Remove unfeatured zip code
  df$zip_code <- NULL
  
  # Make promo features 
  df$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(df$applied_promo_codes))) # 0 is not used, 1 if applied 
  df$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(df$applied_promo_codes)))
  df$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(df$applied_promo_codes)))
  df$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(df$applied_promo_codes)))
  df$num_promos <- df$promo_FREESHIP + df$promo_NEWUSER + df$promo_SAVE20 + df$promo_WINTER50
  # Remove unfeatured promo codes
  df$applied_promo_codes <- NULL
  
  # Bin Customer Age into categories
  df$customer_age_bin <- cut(df$customer_age,
                             breaks = c(-Inf, 18, 30, 45, 60, Inf), 
                             labels = c("Under18", "18to30", "31to45", "46to60", "Over60"),
                             right = FALSE)
  # Remove unbinned cusomter age
  df$customer_age <- NULL
  
  # Bin account_age_days into categories
  df$account_age_bin <- cut(df$account_age_days,
                            breaks = c(-Inf, 104, 250, 504, Inf),
                            labels = c("New", "Developing", "Established", "Loyal"),
                            right = FALSE)
  # Remove unbinned account_age_days
  df$account_age_days <- NULL
  #df$account_age_days[is.na(df$account_age_days)] <- 0
  
  return(df)
}

datTrain <- data_features(datTrain)
datTest <- data_features(datTest)


# Target Encoding for product_subcategory ---------------------------------

# Calculate return rate per subcategory from TRAINING data only
subcat_means <- datTrain %>%
  group_by(product_subcategory) %>%
  summarise(subcat_return_rate = mean(as.numeric(as.character(returned)), 
                                      na.rm = TRUE))

# Apply to both train and test
datTrain <- left_join(datTrain, subcat_means, by = "product_subcategory")
datTest  <- left_join(datTest,  subcat_means, by = "product_subcategory")

# Handle unseen subcategories in test with global mean
global_mean <- mean(as.numeric(as.character(datTrain$returned)), na.rm = TRUE)
datTest$subcat_return_rate[is.na(datTest$subcat_return_rate)] <- global_mean

# Remove original subcategory column
datTrain$product_subcategory <- NULL
datTest$product_subcategory  <- NULL

# Dummy variables --------------------------------------

# Train
datTrain <- dummy_cols(datTrain, 
                       select_columns = c(
                         "marketing_channel",
                         "product_category",
                         "payment_method",
                         "customer_age_bin",
                         "account_age_bin",
                         "zip_region"
                       ),
                       remove_first_dummy = TRUE, # Avoid dummy variable trap
                       remove_selected_columns = TRUE) # Remove original columns after creating dummies
# Test 
datTest <- dummy_cols(datTest, 
                      select_columns = c(
                        "marketing_channel",
                        "product_category",
                        "payment_method",
                        "customer_age_bin",
                        "account_age_bin",
                        "zip_region"
                      ),
                      remove_first_dummy = TRUE, # Avoid dummy variable trap
                      remove_selected_columns = TRUE) # Remove original columns after creating dummies

#Checking if those with no Loyalty tier are the guest checkout people. They are so I am removing the former  
#table(datTrain$loyalty_tier_None, datTrain$guest_checkout) # hey are so I am removing the former  
#datTrain$loyalty_tier_None <- NULL
#datTest$loyalty_tier_None <- NULL

# Checking if account_age_bin_NA/customer_age_bin_NA are identical to guest_checkout. If so, removing the former. 
table(datTrain$account_age_bin_NA, datTrain$guest_checkout)
table(datTrain$customer_age_bin_NA, datTrain$guest_checkout)

datTrain$account_age_bin_NA <- NULL
datTrain$customer_age_bin_NA <- NULL

datTest$account_age_bin_NA <- NULL
datTest$customer_age_bin_NA <- NULL

sum(table(names(datTest) == names(datTrain))) # Checking if the names of the variables are the same in both datasets. They are.es

# Making NAs 0s after processing -----------------------------

bin_cols <- grep("_bin_", names(datTrain), value = TRUE)
datTrain[, bin_cols][is.na(datTrain[, bin_cols])] <- 0

bin_cols_test <- grep("_bin_", names(datTest), value = TRUE)
datTest[, bin_cols_test][is.na(datTest[, bin_cols_test])] <- 0

sum(is.na(datTrain))  # should return 0
sum(is.na(datTest))   # should return 0


# SMOTE for Class Balance -------------------------------------------------

table(datTrain$returned)
table(datTrain$returned)[2] / nrow(datTrain) # 15 percent of the orders were returned (=1). Moderate degree of imbalance)
prop.table(table(datTrain$returned)) # Shows % of each class. 1 is "Yes" returned. 

# Seperate features and target
X_train <- datTrain[, names(datTrain) != "returned"]
y_train <- as.factor(datTrain$returned)

# Apply SMOTE to the training data
set.seed(42) 
smote_result <- SMOTE(X_train, y_train, K=5, dup_size = 0) # K is the number of nearest neighbors to use when generating synthetic samples. dup_size is the number of times to duplicate the minority class before applying SMOTE. Setting it to 0 means no duplication, and only synthetic samples will be generated.)
datTrain_smote <- smote_result$data
names(datTrain_smote)
table(datTrain_smote$class) # The returned outcome variable was renamed as "class" 

#Rename target column 
which(names(datTrain_smote) == "class") 
names(datTrain_smote)[ncol(datTrain_smote)] <- "returned"

# Checking the new class balance after SMOTE
table(datTrain_smote$returned)
prop.table(table(datTrain_smote$returned)) #53% 0, 47% 1. 

names(datTrain_smote)
names(datTest)

names(datTrain_smote) <- make.names(names(datTrain_smote))
names(datTest) <- make.names(names(datTest))
names(datTrain) <- make.names(names(datTrain))

datTrain_smote$returned <- as.factor(datTrain_smote$returned)



# Standardizing column names ----------------------------------------------

names(datTrain) <- make.names(names(datTrain))

names(datTest) <- make.names(names(datTest))

# Logistic Regression Baseline model  -------------------------------------

m1_lr_full <- glm(returned ~ ., data = datTrain, family = binomial)
summary(m1_lr_full)

# Train AUC
probs_lr_full <- predict(m1_lr_full, newdata = datTrain, type = "response")
roc_lr_full <- roc(datTrain$returned, probs_lr_full, plot = TRUE, grid = TRUE, col = "blue")
auc_lr_full <- auc(roc_lr_full)

# Test AUC
probs_lr_full_test <- predict(m1_lr_full, newdata = datTest, type = "response")
roc_lr_full_test <- roc(datTest$returned, probs_lr_full_test, plot = TRUE, grid = TRUE, col = "red")
auc_lr_full_test <- auc(roc_lr_full_test)

# Results table
results <- data.frame(
  Model = "Baseline Logistic Regression",
  Train_AUC = as.numeric(auc_lr_full),
  Test_AUC = as.numeric(auc_lr_full_test)
)
results
# Variable Selection ------------------------------------------------------

# Lasso variable Selection ==========================

# Prepare matrices 
x_matrix <- as.matrix(datTrain[, !names(datTrain) %in% "returned"])
y_target  <- as.numeric(datTrain$returned)

# Fit Lasso
fitLasso <- glmnet(x = x_matrix,
                   y = y_target,
                   family = "binomial",
                   alpha = 1,
                   standardize = TRUE)

# Checking coefficients at different lambda values
par(mfrow = c(1, 2))
plot(fitLasso, xvar = "lambda", label = TRUE) # smaller the -log(lambda), the larger the lamba and greater penalty --> coefficients go to 0 
plot(fitLasso, xvar = "norm", label = TRUE) # sum of coefficients. The smaller the norm, the greater the penalty and more coefficients go to 0.

length(fitLasso$lambda)
betaHat <- fitLasso$beta # rows are the variables parameters , columns are the different lambda values. The values are the coefficients for each variable at each lambda value.
dim(betaHat) # left columns are bigger lamba (high penalty), gets smaller as it goes to the right. 
# FREESHIP stays is chosen even with high penalty, so it is likely an important variable.
apply(betaHat, 2, function(x) sum(x != 0)) # Shows how many variables are included in the model at each lambda value. The smaller the lambda, the more variables are included. )
beta_last <- betaHat[, ncol(betaHat)]
names(beta_last[beta_last == 0]) # These variables never enter the model, not even at the lowest penalty 
sum(beta_last == 0) # 48 variables have coefficients of 0 at the last lambda value, which is the smallest lambda value and therefore the least penalty.


#CV Lasso =====================

set.seed(42) 
fid <- sample(1:10, size = nrow(datTrain), replace = TRUE)
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)

cvLasso <- cv.glmnet(x = x_matrix, 
                     y = y_target,
                     family = "binomial",
                     alpha = 1, 
                     type.measure = "auc",
                     parallel = TRUE,  # fixed typo: paralle -> parallel
                     nfolds = 10, 
                     foldid = fid)
stopCluster(cl) 

par(mfrow = c(1,1))
plot(cvLasso)

cvLasso$lambda.min
log(cvLasso$lambda.min)
cvLasso$lambda.1se
log(cvLasso$lambda.1se)

coef(cvLasso, s = "lambda.min")
coef(cvLasso, s = "lambda.1se")

# Predict on test data
x_matrix_test <- as.matrix(datTest[, !names(datTest) %in% "returned"])

probs_cvlasso_min <- predict(cvLasso, newx = x_matrix_test, s = "lambda.min", type = "response")
probs_cvlasso_1se <- predict(cvLasso, newx = x_matrix_test, s = "lambda.1se", type = "response")

roc_cvlasso_min <- roc(datTest$returned, as.vector(probs_cvlasso_min), 
                       plot = TRUE, grid = TRUE, col = "green")
auc_cvlasso_min <- auc(roc_cvlasso_min)

roc_cvlasso_1se <- roc(datTest$returned, as.vector(probs_cvlasso_1se), 
                       plot = TRUE, grid = TRUE, col = "orange")
auc_cvlasso_1se <- auc(roc_cvlasso_1se)

# Update results table with actual AUC values
results <- rbind(results, data.frame(
  Model = c("CV Lasso (lambda.min)", "CV Lasso (lambda.1se)"),
  Train_AUC = c(NA, NA),
  Test_AUC = c(as.numeric(auc_cvlasso_min), as.numeric(auc_cvlasso_1se))
))
results

## Decision Trees Analysis -------------------------------------------------

# Convert returned to factor for classification tree
datTrain$returned <- factor(datTrain$returned, levels = c(0,1), labels = c("No", "Yes"))
datTest$returned  <- factor(datTest$returned,  levels = c(0,1), labels = c("No", "Yes"))

# Full decision tree with no pruning
tr1 <- rpart(returned ~., data = datTrain, method = "class", 
             control = rpart.control(cp = 0))

tr1$cptable
# Minimum 10 fold CV error 
min.cv <- which.min(tr1$cptable[, 4])
tr1$cptable[min.cv, 4]

# 1-SE rule 
sel <- which(tr1$cptable[, 4] < tr1$cptable[min.cv, 4] + tr1$cptable[min.cv, 5])
cps <- tr1$cptable[sel, 1]

# Pruned trees
tr_min <- rpart::prune(tr1, cp = tr1$cptable[min.cv, 1])
tr_1se <- rpart::prune(tr1, cp = max(cps))

# Predictions
probs_tr_min <- predict(tr_min, newdata = datTest, type = "prob")
roc_tr_min   <- roc(datTest$returned, probs_tr_min[, 2], plot = TRUE, grid = TRUE)
auc_tr_min   <- auc(roc_tr_min)

probs_tr_1se <- predict(tr_1se, newdata = datTest, type = "prob")
roc_tr_1se   <- roc(datTest$returned, probs_tr_1se[, 2], plot = TRUE, grid = TRUE)
auc_tr_1se   <- auc(roc_tr_1se)

# Update results
results <- rbind(results, data.frame(
  Model = c("Decision Tree (min CV)", "Decision Tree (1SE rule)"),
  Train_AUC = c(NA, NA),
  Test_AUC = c(as.numeric(auc_tr_min), as.numeric(auc_tr_1se))
))
results

# RandomForest  -----------------------------------------------------------

class(datTrain$returned)

# Fixing column names for RF 
colnames(datTrain) <- make.names(colnames(datTrain))
colnames(datTest) <- make.names(colnames(datTest))
colnames(x_matrix) <- make.names(colnames(x_matrix))
colnames(x_matrix_test) <- make.names(colnames(x_matrix_test))

set.seed(42)
rf <- randomForest(returned ~., 
                   data = datTrain,
                   mtry = floor(sqrt(ncol(datTrain)-1)),
                   ntree = 500, 
                   importance = TRUE,
                   na.action = na.roughfix)

# Predictions on test data
probs_rf <- predict(rf, newdata = na.roughfix(datTest), type = "prob")
roc_rf   <- roc(datTest$returned, probs_rf[, 2], plot = TRUE, grid = TRUE)
auc_rf   <- auc(roc_rf)

# Update results - fix column name Model not Models
results <- rbind(results, data.frame(
  Model = "Random Forest",
  Train_AUC = NA,
  Test_AUC = as.numeric(auc_rf)
))
results


plot(roc_rf, grid = c(0.1, 0.1), col = "forestgreen")
plot(roc_tr_min, add = TRUE, col = "red")
plot(roc_tr_1se, add = TRUE, col = "orange")

# plotting OUT OF BAG error 
plot(rf, main = "")
rf
legend("bottomright", legend = c("Overall", "Yes", "No"), lty = c(1, 3, 3), col = c("black", "forestgreen", "red")) # Plot shows its very bad at predicting Yes - prolly because of the imbalance 

# Random Forest on SMOTE data -----------------------------------------------------------
set.seed(42)

rf_sm <- randomForest(returned ~., 
                   data = datTrain_smote,
                   mtry = floor(sqrt(ncol(datTrain_smote)-1)),
                   ntree = 500, 
                   importance = TRUE)

probs_rf_sm <- predict(rf_sm, newdata = datTest, type = "prob")
roc_rf_sm <- roc(datTest$returned, probs_rf_sm[,2], plot = TRUE, grid = TRUE)
auc_rf_sm <- auc(roc_rf_sm)
results <- rbind(results, data.frame(
  Models = "Random Forest SMOTE",
  Train_AUC = NA,
  Test_AUC = auc_rf_sm
))
results

# Confusion matrix 
pred_rf_sm_class <- predict(rf_sm, newdata = datTest, type = "class")
table(pred_rf_sm_class, datTest$returned) 


# XGBoost Model -----------------------------------------------------------

# Making Return Variable a integer for XGBoost
datTrain$returned <- as.integer(datTrain$returned) - 1
datTest$returned  <- as.integer(datTest$returned) - 1


xgb <- xgboost(x = x_matrix,
                y = factor(y_target),          # convert to factor for binary classification
                objective = "binary:logistic",
                learning_rate = 0.05,          # renamed from eta
                min_split_loss = 0,            # renamed from gamma
                max_depth = 4,
                min_child_weight = 1,
                eval_metric = "auc",
                nthread = 4,
                nrounds = 500,
                verbosity = 1)

# Predictions 
xgb.prob <- predict(xgb, x_matrix_test)
roc_xgb <- roc(datTest$returned, xgb.prob, plot = TRUE, grid = TRUE)
auc_xgb <- auc(roc_xgb)
results <- rbind(results, data.frame(
  Model = "XGBoost",
  Train_AUC = NA,
  Test_AUC = as.numeric(auc_xgb)
))
results


# CV XGBoost --------------------------------------------------------------

# Create DMatrix first
dtrain <- xgb.DMatrix(data = x_matrix, label = y_target)

# CV with updated API
xgbCV <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    learning_rate = 0.05,
    min_split_loss = 8.11,
    max_depth = 7,
    min_child_weight = 10,
    eval_metric = "auc",
    nthread = 4
  ),
  data = dtrain,
  nrounds = 500,
  nfold = 10,
  prediction = TRUE,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 10,
  early_stopping_rounds = 10
)
# Getting the best number of rounds based on CV results
best_nrounds <- which.max(xgbCV$evaluation_log$test_auc_mean)
best_nrounds
# Train final model with best nrounds
xgb_final <- xgboost(x = x_matrix,
                     y = factor(y_target),
                     objective = "binary:logistic",
                     learning_rate = 0.05,
                     min_split_loss = 0,
                     max_depth = 4,
                     min_child_weight = 1,
                     nrounds = best_nrounds,
                     verbosity = 0)
# Predictions with final model
xgb_cv_prob <- predict(xgb_final, x_matrix_test)
roc_xgb_cv <- roc(datTest$returned, xgb_cv_prob, plot = TRUE, grid = TRUE)
auc_xgb_cv <- auc(roc_xgb_cv)
results <- rbind(results, data.frame(
  Model = "XGBoost CV",
  Train_AUC = NA,
  Test_AUC = as.numeric(auc_xgb_cv)
))
results 

# Variable importance from xgb-final
importance_matrix <- xgb.importance(model = xgb_final)
xgb.plot.importance(importance_matrix, top_n = 20)
print(importance_matrix)

 # Automated XGBoost Tuning  -----------------------------------------------

# Prepare data
xgb_data <- as.data.frame(x_matrix)
xgb_data$returned <- factor(ifelse(y_target == 1, "Yes", "No"), 
                            levels = c("No", "Yes"))

# Define model spec with tunable parameters
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# CV folds
set.seed(42)
cv_folds <- vfold_cv(xgb_data, v = 10, repeats = 3, strata = returned)

# Random search grid
set.seed(42)
xgb_grid <- grid_random(
  trees(range = c(100, 1000)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(-3, -1)),  # log10 scale: 0.001 to 0.1
  loss_reduction(range = c(0, 10)),
  min_n(range = c(1, 20)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry(range = c(5, ncol(x_matrix))),
  size = 100  # 100 random combinations
)

# Workflow
xgb_workflow <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(returned ~ .)

# Tune - run all together 
library(doParallel)
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
start <- Sys.time()
set.seed(42)
xgb_tuned <- tune_grid(
  xgb_workflow,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE, allow_par = TRUE)
)
stopCluster(cl)
print(Sys.time() - start)

# Best parameters
show_best(xgb_tuned, metric = "roc_auc", n = 5)
select_best(xgb_tuned, metric = "roc_auc")

# Final model with best parameters
xgb_final_spec <- boost_tree(
  trees = 616,
  tree_depth = 6,
  learn_rate = 0.00947,
  loss_reduction = 9.51,
  min_n = 10,
  sample_size = 0.782,
  mtry = 22
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Final workflow
xgb_final_workflow <- workflow() %>%
  add_model(xgb_final_spec) %>%
  add_formula(returned ~ .)

# Train on full training data
xgb_final_fit <- fit(xgb_final_workflow, data = xgb_data)

# Prepare test data
xgb_test_data <- as.data.frame(x_matrix_test)
xgb_test_data$returned <- factor(ifelse(datTest$returned == 1, "Yes", "No"),
                                 levels = c("No", "Yes"))

# Predict on test data
xgb_final_probs <- predict(xgb_final_fit, 
                           new_data = xgb_test_data, 
                           type = "prob")

# AUC
roc_xgb_tuned <- roc(datTest$returned, xgb_final_probs$.pred_Yes,
                     plot = TRUE, grid = TRUE, col = "blue")
auc_xgb_tuned <- auc(roc_xgb_tuned)
auc_xgb_tuned

# Add to results
results <- rbind(results, data.frame(
  Model = "XGBoost Tuned",
  Train_AUC = NA,
  Test_AUC = as.numeric(auc_xgb_tuned)
))
results


# Test Kaggle Data Setup -----------------------------------------------------

# Loading Testing data
test <- read.csv("returns_test.csv", stringsAsFactors = TRUE)

# Save ids before droppping 
test_ids <- test$transaction_id

# Cleaning and adding features 
test_kaggle <- data_cleaning(test)
test_kaggle <- data_features(test_kaggle)

# Target Encoding - join subcat means from trainig
test_kaggle <- left_join(test_kaggle, subcat_means, by = "product_subcategory")
test_kaggle$subcat_return_rate[is.na(test_kaggle$subcat_return_rate)] <- global_mean
test_kaggle$product_subcategory <- NULL

# Dummies 
test_kaggle <- dummy_cols(test_kaggle, 
                          select_columns = c(
                            "marketing_channel",
                            "product_category",
                            "payment_method",
                            "customer_age_bin",
                            "account_age_bin",
                            "zip_region"
                          ),
                          remove_first_dummy = TRUE, 
                          remove_selected_columns = TRUE)

# Remove same columns as training
test_kaggle$account_age_bin_NA <- NULL
test_kaggle$customer_age_bin_NA <- NULL

# Fix NAs in bin columns 
bin_cols_kaggle <- grep("_bin_", names(test_kaggle), value = TRUE)
test_kaggle[, bin_cols_kaggle][is.na(test_kaggle[, bin_cols_kaggle])] <- 0

sum(is.na(test_kaggle))

# Fix column names 
names(test_kaggle) <- make.names(names(test_kaggle))


# Prepare matrix - no returned column in test set
x_matrix_kaggle <- as.matrix(test_kaggle)
colnames(x_matrix_kaggle) <- make.names(colnames(x_matrix_kaggle))

# Make sure column match training
check_columns <- function(train_matrix, test_matrix) {
  train_cols <- colnames(train_matrix)
  test_cols  <- colnames(test_matrix)
  
  missing_from_test  <- setdiff(train_cols, test_cols)
  extra_in_test      <- setdiff(test_cols, train_cols)
  order_matches      <- all(train_cols == test_cols)
  
  cat("Columns in train but missing from test:", 
      ifelse(length(missing_from_test) == 0, "None ✓", paste(missing_from_test, collapse = ", ")), "\n")
  cat("Extra columns in test not in train:", 
      ifelse(length(extra_in_test) == 0, "None ✓", paste(extra_in_test, collapse = ", ")), "\n")
  cat("Column order matches:", ifelse(order_matches, "Yes ✓", "No ✗"), "\n")
}

check_columns(x_matrix, x_matrix_kaggle)

# Check the issue
check_columns(x_matrix, x_matrix_kaggle)

# Fix - align test to train columns
# Add any missing columns as 0, drop any extra columns
missing_cols <- setdiff(colnames(x_matrix), colnames(x_matrix_kaggle))
extra_cols   <- setdiff(colnames(x_matrix_kaggle), colnames(x_matrix))

# Add missing columns filled with 0
for (col in missing_cols) {
  x_matrix_kaggle <- cbind(x_matrix_kaggle, setNames(data.frame(rep(0, nrow(x_matrix_kaggle))), col))
}

# Drop extra columns
x_matrix_kaggle <- x_matrix_kaggle[, !colnames(x_matrix_kaggle) %in% extra_cols]

# Reorder to match training
x_matrix_kaggle <- x_matrix_kaggle[, colnames(x_matrix)]

# Verify
check_columns(x_matrix, x_matrix_kaggle)


# Predict Kaggle Data -----------------------------------------------------

kaggle_probs <- predict(xgb_final, x_matrix_kaggle)

# Check for NAs
sum(is.na(kaggle_probs))

# Create submission 
submission <- data.frame(
  transaction_id = test_ids,
  returned = kaggle_probs
)

head(submission)
nrow(submission)

# Write to CSV
write.csv(submission, file = "submission.csv", row.names = FALSE)
