setwd("~/Documents/STAT 640 Final project")
train <- read.csv("returns_train.csv", stringsAsFactors = FALSE)
test <- read.csv("returns_test.csv", stringsAsFactors = FALSE)
sub <- read.csv("sample_submission.csv", stringsAsFactors = FALSE)

str(test)
str(train)
head(sub)
names(sub)
length(unique(train$zip_code))

library("lubridate")
library("dplyr")
library("stringr")
library("pROC")
#explore the data
names(train)
mean(train$returned) #about 15% of purchases are returned
colSums(is.na(train)) #looking for missing values, customer_age and account_age_days contain many
#implies MNAR, guests checkout and have no account so no account age
table(train$product_category)
table(train$loyalty_tier)
table(train$marketing_channel)
nrow(train)

#feature engineering
train <- train %>%
  mutate(
    transaction_timestamp = ymd_hms(transaction_timestamp),
    hour = hour(transaction_timestamp), #late night purchases could have higher return rates
    weekday = wday(transaction_timestamp, label = TRUE),
    promo_any = ifelse(applied_promo_codes == "" | is.na(applied_promo_codes), 0, 1), #promo code used 1, not used 0
    promo_n = ifelse(applied_promo_codes == "" | is.na(applied_promo_codes), 0, str_count(applied_promo_codes, ",") + 1),
    final_price = price * (1 - discount_pct)
  )
names(train)
# 1 = item returned, 0 = item kept, esimating P(returned = 1|tranaction features)
# given what we know about a purchase, how likely is it to be returned?

#logisitc regression model ----------------------------------------------------
#first prep variables for lr model
train_model <- train %>%
  select(-transaction_id, -customer_id, -transaction_timestamp, -applied_promo_codes)
#convert categorical variables to factors
factor_variables <- c("zip_code", "marketing_channel", "product_category", "product_subcategory", "loyalty_tier", "payment_method", "guest_checkout", "weekday")
train_model[factor_variables] <- lapply(train_model[factor_variables], factor)
#handle missing values in the data
train_model$customer_age[is.na(train_model$customer_age)] <- median(train_model$customer_age, na.rm = TRUE)
train_model$account_age_days[is.na(train_model$account_age_days)] <- median(train_model$account_age_days, na.rm = TRUE)
#convert to factor
train_model$returned <- factor(ifelse(train_model$returned == 1, "Yes", "No"))
str(train_model)
#fit full logistic regression model
log_full <- glm(returned~., data = train_model, family = binomial)
summary(log_full)
#apply stepwise variable selection
log_step <- step(log_full, direction = "both")
summary(log_step)
#backwards selection (BIC)
n <- nrow(train_model)
log_bic <- step(log_full, direction = "backward", k = log(n))
summary(log_bic)
train_model$returned <- factor(train$returned, levels = c(0,1), labels = c("No", "Yes"))
levels(train_model$returned)
table(train_model$returned)
