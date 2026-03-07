install.packages("plotrix")
install.packages("fastDummies")
install.packages("corrplot")
install.packages("pROC")
install.packages("glmnet")
install.packages("doParallel")
install.packages("rpart")
install.packages("smotefamily")
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

# Loading data and checking out structure ---------------------------------
dat <- read.csv("returns_train.csv", stringsAsFactors = TRUE)
# Checking out the structure of the data 
dim(dat)
str(dat) # Returned = 1, kept = 0 

# Checking out the levels in a few variables 
levels(dat$applied_promo_codes)
levels(dat$loyalty_tier)
levels(dat$payment_method)


# Checking Missing values -------------------------------------------------

#Checking how many NAs we have. # NAs in account_age_days are for those people who have checked out as GUESTS 
colSums(is.na(dat)) 
# Customer age and account_age_days have NAs. The numbers are the same for both variables (10491)

# Checking if the NA's of account_age_days match the NA's of customer_age. If so, it's because they are checking out as GUESTS and that info is not available.  
dat[is.na(dat$account_age_days), "customer_age"] # All NAs in account_age_days are for customers with NA in customer_age. This makes sense because they are guests and we don't have their information.
#We keep this NAs as they are because they are informative. 
#Missing at Random = the missingness is explained by other variables. 
      #in this case customer_age is missing because account_days_age is missing 
      #and checking out as guests does not allow for this info to be included 
# After the data is split, dummy variable can be created to say that if NA in account_age_days, then guest is 1. 
   # Turns out there is a variable for this already - guest_checkout. 


# Visualizing class balance -----------------------------------------------

barplot(table(dat$return), main = "Balance of the outcome variable", xlab = "Return", ylab = "Frequency")

# Checking the percentage of returns in the data 
sum(dat$return == 1) / nrow(dat) # 15 percent of the orders were returned (=1). Moderate degree of imbalance 

# Because there is moderate imbalance in the dataset, we can do undersampling or oversampling
# Understampling is remove random samples from the majority class 
# Oversampling creates additional copies of the minority class 
       #SMOTE (Synthetic Minority Over-sampling Technique) is a popular oversampling technique that creates synthetic samples of the minority class.
#First split the data, create dummy variables and then do SMOTE on training data only. Not really super necessary since our split seems to be okay. Optional


# Split the data ----------------------------------------------------------

set.seed(42)
train <- sample(1:nrow(dat), nrow(dat)*0.7)
datTrain <- dat[train, ]
datTest <- dat[-train, ]


# Check for correlated variables  -----------------------------------------

# Checking the correlation in NAs between customer_age and guest_checkout. The gust_checkout variable is a binary variable that indicates whether the customer checked out as a guest (1) or not (0). If the customer checked out as a guest, then we don't have information about their age, which is why we see NAs in the customer_age variable for those cases._
table(is.na(datTrain$customer_age), datTrain$guest_checkout, 
      dnn = c("customer_age is NA", "guest_checkout"))

table(is.na(datTrain$account_age_days), datTrain$guest_checkout, 
      dnn = c("customer_age is NA", "guest_checkout"))


# Make feature ----------------------------------------

str(datTrain)

#======================================================================
# Feature engineering for timestamp variable

# Make timestamp a date format 
# Train
datTrain$transaction_timestamp <- ymd_hms(as.character(datTrain$transaction_timestamp))
class(datTrain$transaction_timestamp)
# Test 
datTest$transaction_timestamp <- ymd_hms(as.character(datTest$transaction_timestamp))
class(datTest$transaction_timestamp)

# Add feature variables for the timestamp 
# Train
datTrain$hour_of_day <- hour(datTrain$transaction_timestamp)
datTrain$day_of_week <- wday(datTrain$transaction_timestamp, label = FALSE) #1 is sun, 7 is sat
datTrain$month <- month(datTrain$transaction_timestamp)
datTrain$is_weekend <- as.integer(wday(datTrain$transaction_timestamp) %in% c(1, 7)) #1 if = 1 or 7, 0 otherwise
datTrain$transaction_timestamp <- NULL
# Test 
datTest$hour_of_day <- hour(datTest$transaction_timestamp)
datTest$day_of_week <- wday(datTest$transaction_timestamp, label = FALSE)
datTest$month <- month(datTest$transaction_timestamp)
datTest$is_weekend <- as.integer(wday(datTest$transaction_timestamp) %in% c(1,7))
datTest$transaction_timestamp <- NULL

str(datTrain[, c("hour_of_day", "day_of_week", "month", "is_weekend")])
str(datTest[, c("hour_of_day", "day_of_week", "month", "is_weekend")])

#==================================================================

# Feature engineer customer age 

# Check distribution and range for ages for abnormalities 
summary(datTrain$customer_age) # negative values for ages - data quality issue 
hist(datTrain$customer_age, breaks = 20, main = "Customer Age Distribution")
sum(datTrain$customer_age < 0, na.rm = TRUE) # Only 25 out of 10491 NAs are negative values. We can set these to NA because they are likely data entry errors.
range(datTrain$customer_age, na.rm = TRUE)
table(datTrain$customer_age[datTrain$customer_age > 0]) # This shows there are entries less than 18. Count them 
sum(datTrain$customer_age < 13, na.rm = TRUE) # 519 rows less than 13 years old. 1256 rows for less than 18 years old. 
sum(datTrain$customer_age == 6, na.rm = TRUE) # Counting how many rows for each age less than 18. Cutoff for real ages is 13. 


# Set negative values and values less than 13 to NA.
# Train
datTrain$customer_age[datTrain$customer_age < 13] <- NA 
# Test 
datTest$customer_age[datTest$customer_age < 13] <- NA

# Bin ages for Lasso ONLY 
# Train
datTrain$customer_age_bin <- cut(datTrain$customer_age,
                                 breaks = c(-Inf, 18, 30, 45, 60, Inf), 
                                 labels = c("Under 18", "18-30", "31-45", "46-60", "Over 60"),
                                 right = FALSE)
table(datTrain$customer_age_bin, useNA = "always") # NAs are guests and people less than 13 y/o 
datTrain$customer_age <- NULL
# Test
datTest$customer_age_bin <- cut(datTest$customer_age,
                                breaks = c(-Inf, 18, 30, 45, 60, Inf), 
                                labels = c("Under 18", "18-30", "31-45", "46-60", "Over 60"),
                                right = FALSE)
table(datTest$customer_age_bin, useNA = "always") # NAs are guests  
datTest$customer_age <- NULL

# =======================================

# Remove the account_age_bins since they are highly correlated with loyalty tier and they are not in the test data.
#table(datTrain$loyalty_tier, datTrain$account_age_bin)
# Checking the correlation between the two variables. Here I had to temporarity make that account_age_bin since I deleted it during the dummy process 
#account_age_bin_temp <- cut(dat$account_age_days,
                            #breaks = c(-Inf, 104, 250, 504, Inf),
                            #labels = c("New", "developing", "established", "loyal"),
                            #right = FALSE)
#Checking correlation 
#table(dat$loyalty_tier, account_age_bin_temp)
#str(datTrain)

# Train
datTrain$account_age_days <- NULL
# Test 
datTest$account_age_days <- NULL

#=====================================

# Promo Code features 
#Train
levels(datTrain$applied_promo_codes) # This shows how there are only 4 levels, but they are used in many combinations 
datTrain$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(datTrain$applied_promo_codes))) # 0 is not used, 1 if applied 
datTrain$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(datTrain$applied_promo_codes)))
datTrain$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(datTrain$applied_promo_codes)))
datTrain$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(datTrain$applied_promo_codes)))
datTrain$applied_promo_codes <- NULL
#Test
levels(datTest$applied_promo_codes)
datTest$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(datTest$applied_promo_codes))) # 0 is not used
datTest$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(datTest$applied_promo_codes)))
datTest$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(datTest$applied_promo_codes)))
datTest$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(datTest$applied_promo_code)))
datTest$applied_promo_codes <- NULL

# High Cat Var Cleaning --------------------------------------------------

#==============================

# Payment method cleaning 
unique(datTrain$payment_method) # Shows that there are Paypal and paypal or Visa and visa. Needs to be fixed
levels(datTest$payment_method) # Same issue with the test data.
# Standardize to lower case
# Train
datTrain$payment_method <- factor(tolower(as.character(datTrain$payment_method)))
levels(datTrain$payment_method) # Now we have only paypal and visa.
# Test 
datTest$payment_method <- factor(tolower(as.character(datTest$payment_method)))
levels(datTest$payment_method) # Same for test data.)
# Merge credit card - visa as visa 
# Train
datTrain$payment_method <- recode(datTrain$payment_method, "credit card - visa" = "visa")
levels(datTrain$payment_method) 
# Test 
datTest$payment_method <- recode(datTest$payment_method, "credit card - visa" = "visa")
levels(datTest$payment_method)

#==============================

# Zip code cleaning 
length(unique(datTrain$zip_code))
table(datTrain$product_subcategory)
table(datTrain$product_category, datTrain$product_subcategory)

sort(unique(datTrain$zip_code))
datTrain$zip_region <- cut(as.integer(substr(datTrain$zip_code, 1, 1)),
                           breaks = c(-Inf, 2, 4, 6, 7, Inf),
                           labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                           right = TRUE)
datTrain$zip_code <- NULL
table(datTrain$zip_region)

datTest$zip_region <- cut(as.integer(substr(datTest$zip_code, 1, 1)),
                          breaks = c(-Inf, 2, 4, 6, 7, Inf),
                          labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                          right = TRUE)

table(datTest$zip_region)
datTest$zip_code <- NULL

#==============================

# Dropping variables that are too complex or not useful 
datTrain$product_subcategory <- NULL
datTest$product_subcategory <- NULL

# Removing variables that are not useful for prediction 
# Train
datTrain$transaction_id <- NULL
datTrain$customer_id <- NULL
# Test
datTest$transaction_id <- NULL
datTest$customer_id <- NULL

# Dummy variables for cat variables  --------------------------------------

# ================================

# Guest checkout variable

# Making the guest_checkout variable from FALSE/TRUE to 0/1 
#Train
datTrain$guest_checkout <- as.integer(datTrain$guest_checkout == "True") 
# Test
datTest$guest_checkout <- as.integer(datTest$guest_checkout == "True")

# ==============================

# Make the dummies for everything else. 
# Train
datTrain <- dummy_cols(datTrain, 
                       select_columns = c(
                         "marketing_channel",
                         "product_category",
                         "loyalty_tier",
                         "payment_method",
                         "customer_age_bin",
                         "zip_region"
                       ),
                       remove_first_dummy = TRUE, # Avoid dummy variable trap
                       remove_selected_columns = TRUE) # Remove original columns after creating dummies
# Test 
datTest <- dummy_cols(datTest, 
                      select_columns = c(
                        "marketing_channel",
                        "product_category",
                        "loyalty_tier",
                        "payment_method",
                        "customer_age_bin",
                        "zip_region"
                      ),
                      remove_first_dummy = TRUE, # Avoid dummy variable trap
                      remove_selected_columns = TRUE) # Remove original columns after creating dummies

str(datTest)

#Checking if those with no Loyalty tier are the guest checkout people. They are so I am removing the former  
table(datTrain$loyalty_tier_None, datTrain$guest_checkout) # hey are so I am removing the former  
datTrain$loyalty_tier_None <- NULL
datTest$loyalty_tier_None <- NULL

str(datTrain)      

sum(table(names(datTest) == names(datTrain))) # Checking if the names of the variables are the same in both datasets. They are.es

#====================

# Cleaning Customer AGE = NAs from making ages < 13 as NA. They should be 0 
# Train
age_bin_cols <- grep("customer_age_bin", names(datTrain), value = TRUE)
datTrain[, age_bin_cols][is.na(datTrain[, age_bin_cols])] <- 0
# Test
age_bin_cols_test <- grep("customer_age_bin", names(datTest), value = TRUE)
datTest[, age_bin_cols_test][is.na(datTest[, age_bin_cols_test])] <- 0

sum(is.na(datTrain))  # should return 0
sum(is.na(datTest))   # should return 0

# Correlation  --------------------------------------------------------------------
sum(datTrain$returned == 1) / nrow(datTrain)
str(datTrain)


# Find the correlation matrix for the numeric variables except the outcome variable - determins redundancy in the variables 
feature_matrix <- cor(datTrain[, sapply(datTrain, is.numeric) & names(datTrain) != "returned"],
                      use = "pairwise.complete.obs")

corrplot(feature_matris, method = "color", tl.cex = 0.6)

# Finds correlations with target variable
target_cor <- cor(datTrain[, sapply(datTrain, is.numeric)],
                  use = "pairwise.complete.obs")[, "returned"]
sort(target_cor, decreasing = TRUE)


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
which(names(datTrain_smote) == "class") # column 32, the last column, is the class column 
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

# Logistic Regression Baseline model  -------------------------------------

m1_lr_full <- glm(returned ~ ., data = datTrain, family = binomial)

summary(m1_lr_full) # Tells me which variables are significant based on p < 0.05 

# Make predictions to find the AUC
probs_lr_full <- predict(m1_lr_full, newdata = datTrain, type = "response") # Probs that return = 1 given the model 
roc_lr_full <- roc(datTrain$returned, probs_lr_full, plot = TRUE, grid = TRUE, col = "blue", main = "ROC Curve for Logistic Regression Full Model")
auc_lr_full <- auc(roc_lr_full) # 0.713

pros_lr_full_test <- predict(m1_lr_full, newdata = datTest, type = "response")
roc_lr_full_test <- roc(datTest$returned, pros_lr_full_test, plot = TRUE, grid = TRUE, col = "red", main = "ROC Curve for Logistic Regression Full Model")
auc_lr_full_test <- auc(roc_lr_full_test) # 0.7144.

lr_full_auc <- c(Train = 0.713, Test = 0.7144)

results <- data.frame(
  Model = c("Baseline Logistic Regression"),
  Train_AUC = c(0.713),
  Test_AUC = c(0.7144)
)
results

# Variable Selection ------------------------------------------------------

#==========================
# Lasso variable Selection 
#==========================

# Prepare data - glmnet requires matrix format for predictors 
datTrain['y'] <- datTrain$returned
sum(datTrain['y'] != datTrain["returned"]) 
datTrain$returned <- NULL

datTest['y'] <- datTest$returned
sum(datTest['y'] != datTest["returned"])
datTest$returned <- NULL

names(datTrain == "returned")
names(datTest == "returned")

x_matrix <- as.matrix(datTrain[, 1:(ncol(datTrain)-1)])
y_target <- datTrain$y

# fit model with lasso penalty alpha = 1

fitLasso <- glmnet(x = x_matrix, 
                   y = y_target,
                   family = "binomial", 
                   alpha = 1, 
                   standardize = TRUE)

par(mfrow = c(1, 2))
plot(fitLasso, xvar = "lambda", label = TRUE) # smaller the -log(lambda), the larger the lamba and greater penalty --> coefficients go to 0 
plot(fitLasso, xvar = "norm", label = TRUE) # sum of coefficients. The smaller the norm, the greater the penalty and more coefficients go to 0.

length(fitLasso$lambda)
betaHat <- fitLasso$beta # rows are the variables parameters , columns are the different lambda values. The values are the coefficients for each variable at each lambda value.
dim(betaHat) # left columns are bigger lamba (high penalty), gets smaller as it goes to the right. 
# FREESHIP stays is chosen even with high penalty, so it is likely an important variable.
apply(betaHat, 2, function(x) sum(x != 0)) # Shows how many variables are included in the model at each lambda value. The smaller the lambda, the more variables are included. )
beta_last <- betaHat[, ncol(betaHat)]
names(beta_last[beta_last == 0])
sum(beta_last == 0) # 48 variables have coefficients of 0 at the last lambda value, which is the smallest lambda value and therefore the least penalty.

#=====================
# CV Lasso 
#=====================

set.seed(42) 
fid <- sample(1:10, size = nrow(datTrain), replace = TRUE) # Create 10 folds for cross validation. Each row is randomly assigned a number from 1 to 10.))
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)

cvlasso <- cv.glmnet(x = x_matrix, 
                     y = y_target,
                     family = "binomial",
                     alpha = 1, 
                     type.measure = "auc",
                     paralle = TRUE, 
                     nfolds = 10, foldid = fid)
stopCluster(cl) 
par(mfrow = c(1,1))
plot(cvlasso)
# smallest lambda value 
cvlasso$lambda.min
log(cvlasso$lambda.min) # corresponds to the most right vertical line 

# Lambda selected by one standard deviation rule
cvlasso$lambda.1se
log(cvlasso$lambda.1se) # corresponds to the middle vertical line
coef(cvlasso, s = "lambda.min") # Shows the coefficients for each variable at the lambda value that minimizes the mean cross-validated error. 17 variables have non-zero coefficients at this lambda value.sso)
coef(cvlasso, s = "lambda.1se") # Shows the coefficients for each variable at the lambda value that is one standard error above the minimum. 10 variables have non-zero coefficients at this lambda value.

# Predicting using the cvlasso
levels(as.factor(datTest$y))
ncol(datTest[, -ncol(datTest)]) # testdata with only the predictors  
x_test <- as.matrix(datTest[, -ncol(datTest)])
probs_cvlasso_min <- predict(cvlasso, newx = x_test, s = "lambda.min", type = "response")
probs_cvlasso_1se <- predict(cvlasso, newx = x_test, s = "lambda.1se", type = "response")

roc_cvlasso_min <- roc(datTest$y, as.vector(probs_cvlasso_min), plot = TRUE, grid = TRUE, col = "green", main = "ROC Curve for CV Lasso Model")
auc(roc_cvlasso_min) # 0.7138
roc_cvlasso_1se <- roc(datTest$y, as.vector(probs_cvlasso_1se), plot = TRUE, grid = TRUE, col = "orange", main = "ROC Curve for CV Lasso Model")
auc(roc_cvlasso_1se) #0.7079

results <- data.frame(
  Models = c('Baseline Logistic Regression', 'CV Lasso (lambda.min)', 'CV Lasso (lambda.1se)'),
  Train_AUC = c(0.713, NA, NA),
  Test_AUC = c(0.7144, 0.7138, 0.7079)
)
results


# Decision Trees Analysis -------------------------------------------------

# Reloading the data and splitting it just for decision trees preprocessing 
dat <- read.csv("returns_train.csv", stringsAsFactors = TRUE)
set.seed(42)
train <- sample(1:nrow(dat), nrow(dat)*0.7)
datTrain_tree <- dat[train, ]
datTest_tree  <- dat[-train, ]

# Preprocessing function for decision trees

preprocess_tree <- function(df) {
  #Drop ID columns 
  df$transaction_id <- NULL
  df$customer_id <- NULL
  
  #Timempstamps features
  df$transaction_timestamp <- ymd_hms(as.character(df$transaction_timestamp))
  df$hour_of_day <- hour(df$transaction_timestamp)
  df$day_of_week <- wday(df$transaction_timestamp, label = FALSE) 
  df$month <- month(df$transaction_timestamp)
  df$is_weekend <- as.integer(wday(df$transaction_timestamp) %in% c(1,7))
  df$transaction_timestamp <- NULL
  
  # Fix customer_age - remove invalid ages
  df$customer_age[df$customer_age < 13] <- NA
  
  # Drop account_age_days - perfectly correlated with loyatly tier 
  df$account_age_days <- NULL
  
  # Fix payment_method casing and merge visa variands
  df$payment_method <- factor(tolower(as.character(df$payment_method)))
  df$payment_method <- recode(df$payment_method, "credit card - visa" = "visa")
  
  # Drop product subcategory - too many levels, product category will capture signal
  df$product_subcategory <- NULL
  
  # Promo code features 
  df$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(df$applied_promo_codes)))
  df$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(df$applied_promo_codes)))
  df$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(df$applied_promo_codes)))
  df$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(df$applied_promo_codes)))
  df$applied_promo_codes <- NULL
  
  # Zip region
  df$zip_region <- cut(as.integer(substr(df$zip_code, 1, 1)),
                        breaks = c(-Inf, 2, 4, 6, 7, Inf),
                        labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                        right = TRUE)
  df$zip_code <- NULL
  
  # Make returned a factor for clarrification 
  df$returned <- factor(df$returned, levels = c(0, 1), labels = c("No", "Yes"))
  
  return(df)
}

# Apply processing to both training and test data 
datTrain_tree <- preprocess_tree(datTrain_tree)
datTest_tree <- preprocess_tree(datTest_tree)
str(datTrain_tree)

# Full decision tree with no pruning (cp =0)
tr1 <- rpart(returned~., data = datTrain_tree, method = "class", control = rpart.control(cp = 0))
tr1$cptable
# Minimum 10 fold CV error 
min.cv <- which.min(tr1$cptable[,4])
tr1$cptable[min.cv,4]
# 1-SD deviation rule 
sel <- which(tr1$cptable[,4] < tr1$cptable[min.cv,4] + tr1$cptable[min.cv,5]) 
sel # gives me the indeces of the CV which are less than the minimum CV error + 1 SD. 
tr1$cptable[sel, ] 
cps <- tr1$cptable[min(sel), 1] # Biggest Complexity parameter that is within 1 SD of the minimum CV error. This will create a smaller (less complex tree) that is good to use 
# DT with min CV error 
tr_min <- prune(tr1, cp = tr1$cptable[min.cv, 1])
# DT with 1 SD rule 
tr_1se <- prune(tr1, cp = cps)

# Prdictions with the two pruned trees 
# min CV error tree 
probs_tr_min <- predict(tr_min, newdata = datTest_tree, type = "prob")
probs_tr_min
probs_tr_min[,2]
roc_tr_min <- roc(datTest_tree$returned, probs_tr_min[,2], plot = TRUE, grid = TRUE)
auc_tr_min <- auc(roc_tr_min) # 0.6799
# 1 SD tree
probs_tr_1se <- predict(tr_1se, newdata = datTest_tree, type = "prob")
roc_tr_1se <- roc(datTest_tree$returned, probs_tr_1se[,2], plot = TRUE, grid = TRUE)
auc_tr_1se <- auc(roc_tr_1se)
results <- rbind(results, data.frame(
  Models = c('Decision Tree (min CV error)', 'Decision Tree (1 SD rule)'),
  Train_AUC = c(NA, NA),
  Test_AUC = c(auc_tr_min, auc_tr_1se)
))
results


# RandomForest  -----------------------------------------------------------

class(datTrain_tree$returned)
set.seed(42)

rf <- randomForest(returned ~., 
                   data = datTrain_tree,
                   mtry = floor(sqrt(ncol(datTrain_tree)-1)),
                   ntree = 500, 
                   importance = TRUE,
                   na.action = na.roughfix)

probs_rf <- predict(rf, newdata = na.roughfix(datTest_tree), type = "prob")
roc_rf <- roc(datTest_tree$returned, probs_rf[,2], plot = TRUE, grid = TRUE)
auc_rf <- auc(roc_rf)
results <- rbind(results, data.frame(
  Models = 'Random Forest',
  Train_AUC = NA,
  Test_AUC = auc_rf
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

library(xgboost)

x_train_xgb <- matrix(as.numeric(as.matrix(datTrain_smote[, !names(datTrain_smote) %in% "returned"])),
                      nrow = nrow(datTrain_smote))

colnames(x_train_xgb) <- names(datTrain_smote)[names(datTrain_smote) != "returned"]

y_train_xgb <- as.numeric(as.character(datTrain_smote$returned))

x_test_xgb <- matrix(as.numeric(as.matrix(datTest[, !names(datTest) %in% "returned"])),
                     nrow = nrow(datTest))

colnames(x_test_xgb) <- names(datTest)[names(datTest) != "returned"]

y_test_xgb <- as.numeric(datTest$returned)

# Convert to DMatrix format for XGBoost optimized data structure
dtrain <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)
dtest <- xgb.DMatrix(data = x_test_xgb, label = y_test_xgb)

dim(x_train_xgb)
dim(x_test_xgb)

# XGBoost parameters
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # optimize for AUC
  eta = 0.1,                      # learning rate
  max_depth = 6,                  # tree depth
  subsample = 0.8,                # row sampling per tree
  colsample_bytree = 0.8          # column sampling per tree
)

# Train with cross validation to find optimal number of rounds
set.seed(42)
cv_xgb <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,    # stop if no improvement after 20 rounds
  verbose = 1
)

# Best number of rounds
best_nrounds <- which.max(cv_xgb$evaluation_log$test_auc_mean)
best_nrounds

# Train final model with best number of rounds
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain, 
  nround = best_nrounds, 
  evals = list(train = dtrain, test = dtest),
  verbose = 1
)

probs_xbg <- predict(xgb_model, newdata = dtest, type = "response")
roc_xgb <- roc(y_test_xgb, probs_xbg, plot = TRUE, grid = TRUE)
auc_xgb <- auc(roc_xgb)
results <- rbind(results, data.frame(
  Models = "XGBoost",
  Train_AUC = NA,
  Test_AUC = auc_xgb)
)
results  
