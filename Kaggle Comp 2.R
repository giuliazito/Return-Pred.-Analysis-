
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(plotrix)
library(corrplot)
library(pROC)
library(doParallel)
library(rpart)
library(smotefamily)
library(xgboost)
library(tidymodels)
library(finetune)
library(caret)


# Loading Data ------------------------------------------------------------

dat <- read_csv("returns_train.csv")


# Data Cleaning -----------------------------------------------------------

data_cleaning <- function(df) {
  
  # Remove unique identifier 
  df$transaction_id <- NULL
  
  # Make transaction_timestmap a date format
  if("transaction_timestamp" %in% names(df)) {
    df$transaction_timestamp <- lubridate::ymd_hms(df$transaction_timestamp)
  }
  
  # Make customer_age absolute factors
  if("customer_age" %in% names(df)) {
    df$customer_age <- abs(df$customer_age)
  }
  
  # Fix payment method 
  if("payment_method" %in% names(df)) {
    df$payment_method <- tolower(trimws(df$payment_method))
    df$payment_method <- recode(df$payment_method,"credit card - visa" = "visa")
  }
  
  return(df)
}

dat <- data_cleaning(dat)

# Features Variables ------------------------------------------------------

feature_variables <- function(df) {
  
  # Extract promo code length to see which transactions used none, 1 or 2 
  df$promo_order_length <- str_length(df$applied_promo_codes) 
  df$applied_promo_codes <- NULL
  
  # Extract time, day, month, year of timestamps 
  df$time <- as.integer(format(df$transaction_timestamp, "%H%M%S"))
  df$day <- as.integer(format(df$transaction_timestamp, "%d"))
  df$month <- as.integer(format(df$transaction_timestamp, "%m"))
  df$year <- as.integer(format(df$transaction_timestamp, "%Y"))
  df$transaction_timestamp <- NULL
  
  return(df)
  
}

dat <- feature_variables(dat)

# Make Factors  -----------------------------------------------------------

make_factors <- function(df) {
  
  # Make variables into factors 
  df$marketing_channel <- as.factor(df$marketing_channel)
  df$product_category <- as.factor(df$product_category)
  df$product_subcategory <- as.factor(df$product_subcategory)
  df$payment_method <- as.factor(df$payment_method)
  df$guest_checkout <- as.factor(df$guest_checkout)
  df$promo_order_length <- factor(df$promo_order_length, 
                                levels = sort(unique(df$promo_order_length)))
  df$loyalty_tier <- factor(df$loyalty_tier, 
                          levels = c("None", "Bronze", "Silver", "Gold", "Platinum"))
  
  return(df)
}

dat <- make_factors(dat)



# Splitting Data ----------------------------------------------------------

# 80/20 data split 

set.seed(42)

train <- createDataPartition(dat$returned, p = 0.8, list = FALSE)
datTrain <- dat[train, ]
datTest <- dat[-train, ]

dim(datTrain)
dim(datTest)


# XGBoost Training --------------------------------------------------------

# Define matrices 
train_x <- data.matrix(datTrain[, -which(names(datTrain) == "returned")])
train_y <- datTrain$returned

test_x <- data.matrix(datTest[, -which(names(datTest) == "returned")])
test_y <- datTest$returned

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data= test_x, label = test_y)

# Define watchlist
watchlist = list(train = xgb_train, test = xgb_test)

# Fit model
xgb_model = xgb.train(data = xgb_train, 
                      learning_rate = 0.4,
                      max.depth = 3, 
                      watchlist = watchlist,
                      nrounds = 1000, 
                      objective = "binary:logistic", 
                      eval_metric = "auc")

eval_log <- attributes(xgb_model)$evaluation_log
best_round <- which.max(eval_log$test_auc)
print(paste("Best round:", best_round,
            "Test AUC:", eval_log$test_auc[best_round]))

# Predict
pred_xgb <-predict(xgb_model, newdata = xgb_test, iterationrange = c(1, best_round +1))

roc_xgb <- roc(test_y, pred_xgb, quiet = TRUE)
print(auc(roc_xgb))


# Loading Kaggle Test Data  -----------------------------------------------

kaggle <- read_csv("returns_test.csv")
test_id <- kaggle$transaction_id

# Cleaning and processing kaggle data

kaggle <- data_cleaning(kaggle)
kaggle <- feature_variables(kaggle)
kaggle <- make_factors(kaggle)

# Define matrices

names(kaggle)
kaggle_x <- data.matrix(kaggle)

xgb_kaggle_test <- xgb.DMatrix(data = kaggle_x)

# Predict on kaggle data

kaggle_preds <- predict(xgb_model, newdata = xgb_kaggle_test, iterationrange = c(1, best_round + 1))

submission <- data.frame(transaction_id = test_id, 
                         returned = kaggle_preds)

write.csv(submission, "submission_19.csv", row.names = FALSE)
