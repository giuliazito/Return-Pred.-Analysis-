library(tidyverse)
install.packages("plotrix")
library(plotrix)


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
install.packages("tidyverse")
library(tidyverse)

# Make timestamp a date format 
datTrain$transaction_timestamp <- ymd_hms(as.character(datTrain$transaction_timestamp))
head(datTrain$transaction_timestamp)
class(datTrain$transaction_timestamp)

datTest$transaction_timestamp <- ymd_hms(as.character(datTest$transaction_timestamp))
head(datTest$transaction_timestamp)
class(datTest$transaction_timestamp)

# Add feature variables for the timestamp 
datTrain$hour_of_day <- hour(datTrain$transaction_timestamp)
datTrain$day_of_week <- wday(datTrain$transaction_timestamp, label = FALSE) #1 is sun, 7 is sat
datTrain$month <- month(datTrain$transaction_timestamp)
datTrain$is_weekend <- as.integer(wday(datTrain$transaction_timestamp) %in% c(1, 7)) #1 if = 1 or 7, 0 otherwise

datTest$hour_of_day <- hour(datTest$transaction_timestamp)
datTest$day_of_week <- wday(datTest$transaction_timestamp, label = FALSE)
datTest$month <- month(datTest$transaction_timestamp)
datTest$is_weekend <- as.integer(wday(datTest$transaction_timestamp) %in% c(1,7))

# Drop the timestamp variable
datTrain$transaction_timestamp <- NULL
str(datTrain[, c("hour_of_day", "day_of_week", "month", "is_weekend")])

datTest$transaction_timestamp <- NULL
str(datTest[, c("hour_of_day", "day_of_week", "month", "is_weekend")])

# Bin costumer age and account_age_days 

summary(datTrain$customer_age) # negative values for ages - data quality issue 
hist(datTrain$customer_age, breaks = 20, main = "Customer Age Distribution")
# check number of negative values in customer_age
sum(datTrain$customer_age < 0, na.rm = TRUE) # Only 25 out of 10491 NAs are negative values. We can set these to NA because they are likely data entry errors.
range(datTrain$customer_age, na.rm = TRUE)
table(datTrain$customer_age[datTrain$customer_age > 0]) # This shows there are entries less than 18. Count them 
sum(datTrain$customer_age < 13, na.rm = TRUE) # 519 rows less than 13 years old. 1256 rows for less than 18 years old. 
sum(datTrain$customer_age == 6, na.rm = TRUE) # Counting how many rows for each age less than 18. Cutoff for real ages is 13. 

# Check distribution of account_age_days 
summary(datTrain$account_age_days)
hist(datTrain$account_age_days, breaks = 20, main = "Account Age Days Distribution") #Skewed but okay. 

# Set negative values and values less than 13 to NA.]
datTrain$customer_age[datTrain$customer_age < 13] <- NA 

datTest$customer_age[datTest$customer_age < 13] <- NA

# Bin ages 
datTrain$customer_age_bin <- cut(datTrain$customer_age,
                                 breaks = c(-Inf, 18, 30, 45, 60, Inf), 
                                 labels = c("Under 18", "18-30", "31-45", "46-60", "Over 60"),
                                 right = FALSE)
table(datTrain$customer_age_bin, useNA = "always") # NAs are guests and people less than 13 y/o 

datTest$customer_age_bin <- cut(datTest$customer_age,
                                breaks = c(-Inf, 18, 30, 45, 60, Inf), 
                                labels = c("Under 18", "18-30", "31-45", "46-60", "Over 60"),
                                right = FALSE)
table(datTest$customer_age_bin, useNA = "always") # NAs are guests  

# Bin account days 
datTrain$account_age_bin <- cut(datTrain$account_age_days,
                                breaks = c(-Inf, 104, 250, 504, Inf), 
                                labels = c("New", "developing", "established", "loyal"),
                                right = FALSE)
table(datTrain$account_age_bin, useNA = "always") # NAs are guests. 

datTest$account_age_bin <- cut(datTest$account_age_days,
                               breaks = c(-Inf, 104, 250, 504, Inf), 
                               labels = c("New", "developing", "established", "loyal"),
                               right = FALSE)
table(datTest$account_age_bin, useNA = "always") # NAs are guests

datTrain$customer_age <- NULL
datTrain$account_age_days <- NULL

datTest$customer_age <- NULL
datTest$account_age_days <- NULL

# Making dummy variables for categorical variables with high number of cats. --------------------------------------------------

unique(datTrain$payment_method) # Shows that there are Paypal and paypal or Visa and visa. Needs to be fixed
levels(datTest$payment_method) # Same issue with the test data.
# Standardize to lower case
datTrain$payment_method <- factor(tolower(as.character(datTrain$payment_method)))
levels(datTrain$payment_method) # Now we have only paypal and visa.

datTest$payment_method <- factor(tolower(as.character(datTest$payment_method)))
levels(datTest$payment_method) # Same for test data.)
# Merge credit card - visa as visa 
datTrain$payment_method <- recode(datTrain$payment_method, "credit card - visa" = "visa")
levels(datTrain$payment_method) 

datTest$payment_method <- recode(datTest$payment_method, "credit card - visa" = "visa")
levels(datTest$payment_method)

length(unique(datTrain$zip_code))
table(datTrain$product_subcategory)
table(datTrain$product_category, datTrain$product_subcategory)
# Dropping subcategory as it has too many levels. Avoid complexity 
datTrain$product_subcategory <- NULL
datTest$product_subcategory <- NULL


sort(unique(datTrain$zip_code))
datTrain$zip_region <- cut(as.integer(substr(datTrain$zip_code, 1, 1)),
                           breaks = c(-Inf, 2, 4, 6, 7, Inf),
                           labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                           right = TRUE)

table(datTrain$zip_region)

datTest$zip_region <- cut(as.integer(substr(datTest$zip_code, 1, 1)),
                          breaks = c(-Inf, 2, 4, 6, 7, Inf),
                          labels = c("Northeast", "Southeast", "Midwest", "South", "West"),
                          right = TRUE)

table(datTest$zip_region)

# Then drop zip_code
datTrain$zip_code <- NULL

datTest$zip_code <- NULL

# Checking promo_codes levels 
levels(datTrain$applied_promo_codes) # This shows how there are only 4 levels, but they are used in many combinations 
datTrain$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(datTrain$applied_promo_codes))) # 0 is not used, 1 if applied 
datTrain$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(datTrain$applied_promo_codes)))
datTrain$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(datTrain$applied_promo_codes)))
datTrain$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(datTrain$applied_promo_codes)))

datTrain$applied_promo_codes <- NULL

levels(datTest$applied_promo_codes)
datTest$promo_FREESHIP  <- as.integer(grepl("FREESHIP",  as.character(datTest$applied_promo_codes))) # 0 is not used
datTest$promo_NEWUSER   <- as.integer(grepl("NEWUSER",   as.character(datTest$applied_promo_codes)))
datTest$promo_SAVE20    <- as.integer(grepl("SAVE20",    as.character(datTest$applied_promo_codes)))
datTest$promo_WINTER50  <- as.integer(grepl("WINTER50",  as.character(datTest$applied_promo_code)))

datTest$applied_promo_codes <- NULL

str(datTrain)

datTrain$transaction_id <- NULL
datTrain$customer_id <- NULL

datTest$transaction_id <- NULL
datTest$customer_id <- NULL

# Dummy variables for cat variables  --------------------------------------

# Making the guest_checkout variable from FALSE/TRUE to 0/1 
datTrain$guest_checkout <- as.integer(datTrain$guest_checkout == "True") 

datTest$guest_checkout <- as.integer(datTest$guest_checkout == "True")

# Make the dummies for everything else. 
install.packages("fastDummies")
library(fastDummies)

datTrain <- dummy_cols(datTrain, 
                       select_columns = c(
                         "marketing_channel",
                         "product_category",
                         "loyalty_tier",
                         "payment_method",
                         "customer_age_bin",
                         "account_age_bin",
                         "zip_region"
                       ),
                       remove_first_dummy = TRUE, # Avoid dummy variable trap
                       remove_selected_columns = TRUE) # Remove original columns after creating dummies
str(datTrain)

datTest <- dummy_cols(datTest, 
                      select_columns = c(
                        "marketing_channel",
                        "product_category",
                        "loyalty_tier",
                        "payment_method",
                        "customer_age_bin",
                        "account_age_bin",
                        "zip_region"
                      ),
                      remove_first_dummy = TRUE, # Avoid dummy variable trap
                      remove_selected_columns = TRUE) # Remove original columns after creating dummies

str(datTest)

#Noticed that there is loyalty tier and account_age_bin which say the same thing 
table(datTrain$loyalty_tier, datTrain$account_age_bin)
# Checking the correlation between the two variables. Here I had to temporarity make that account_age_bin since I deleted it during the dummy process 
account_age_bin_temp <- cut(dat$account_age_days,
                            breaks = c(-Inf, 104, 250, 504, Inf),
                            labels = c("New", "developing", "established", "loyal"),
                            right = FALSE)
#Checking correlation 
table(dat$loyalty_tier, account_age_bin_temp)
str(datTrain)
# Renaming the NAs in customer_age_bin because its the same as guest checkout plus some people less than 13 y/o.
table(datTrain$customer_age_bin_NA, datTrain$guest_checkout)
datTrain$is_unknown_age <- datTrain$customer_age_bin_NA

table(datTest$customer_age_bin_NA, datTest$guest_checkout)
datTest$is_unknown_age <- datTest$customer_age_bin_NA
# Removing the customer_age_bin_NA since its now renamed and taken care for by the guest_checkout variable.
datTrain$customer_age_bin_NA <- NULL 

datTest$customer_age_bin_NA <- NULL

#Checking if those with no Loyalty tier are the guest checkout people. They are so I am removing the former  
table(datTrain$loyalty_tier_None, datTrain$guest_checkout)
datTrain$loyalty_tier_None <- NULL
datTest$loyalty_tier_None <- NULL

str(datTrain)      

age_bin_cols <- grep("customer_age_bin", names(datTrain), value = TRUE) # Calling out all the variables with "customer_age_bin" in the title 
datTrain[, age_bin_cols][is.na(datTrain[, age_bin_cols])] <- 0 # finds all NA positions within those 4 columns and replaced them with 0 
sum(is.na(datTrain))

age_bin_cols_test <- grep("customer_age_bin", names(datTest), value = TRUE) # Calling out all the variables with "customer_age_bin" in the titl
datTest[, age_bin_cols_test][is.na(datTest[, age_bin_cols_test])] <- 0 # finds all NA positions within those 4 columns and replaced them with 0, a]

table(names(datTest) == names(datTrain)) # Checking if the names of the variables are the same in both datasets. They are.es

names(datTrain)
names(datTest)

# Drops account_age_bin columns from datTest because they are not in datTrain. This is because I dropped the loyalty_tier_None variable which was highly correlated with account_age_bin.
datTest <- datTest[, !grepl("account_age_bin", names(datTest))]
# Make sure column order is the same as in datTrain 
datTest <- datTest[, names(datTrain)]
# Checking 
ncol(datTest) == ncol(datTrain)
names(datTest) == names(datTrain)

# Correlation  --------------------------------------------------------------------
sum(datTrain$returned == 1) / nrow(datTrain)
str(datTrain)

install.packages("corrplot")
library(corrplot)

# Find the correlation matrix for the numeric variables except the outcome variable - determins redundancy in the variables 
feature_matris <- cor(datTrain[, sapply(datTrain, is.numeric) & names(datTrain) != "returned"],
                      use = "pairwise.complete.obs")

corrplot(feature_matris, method = "color", tl.cex = 0.6)
# Only is_unknown_age and guest_checkout are highly correlated enough to point out. 
# is_unknown_age only carries 2% of the the data so it can be dropped. 
datTrain$is_unknown_age <- NULL
datTest$is_unknown_age <- NULL

# Finds correlations with target variable
target_cor <- cor(datTrain[, sapply(datTrain, is.numeric)],
                  use = "pairwise.complete.obs")[, "returned"]
sort(target_cor, decreasing = TRUE)



# Logistic Regression Baseline model  -------------------------------------

m1_lr_full <- glm(returned ~ ., data = datTrain, family = binomial)

summary(m1_lr_full) # Tells me which variables are sinificant based on p < 0.05 

# Make predictions to find the AUC
probs_lr_full <- predict(m1_lr_full, newdata = datTrain, type = "response") # Probs that return = 1 given the model 
# Install pROC package to calculate AUC
install.packages("pROC")
library(pROC)
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
install.packages("glmnet")
library(glmnet)

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
install.packages("doParallel")
library(doParallel)
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


