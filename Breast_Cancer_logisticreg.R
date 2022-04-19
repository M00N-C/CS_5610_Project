### Installing necessary packages
#install.packages("tidyverse")
#install.packages("caTools") # For Logistic regression
#install.packages("ROCR")	 # For ROC curve to evaluate model
#install.packages("pscl")  # Model evaluation


### Loading package
library(plyr)
library(tidyverse)
library(caTools)
library(ROCR)
library(carData)
library(caret)
library(car)
library(pscl)


###load dataset
data_all <- readxl::read_xlsx("Breast Cancer data - CS 5610.xlsx")
#remove ids and standard errors, setting diagnosis to factor variable
# factor set 1 == "M", 0 == "B"
data_all <- data_all[,-1]
data_all <- data_all[, !grepl('_se', colnames(data_all))]
colnames(data_all)[c(9, 19)] <- c("concave_points_mean", "concave_points_worst")
data_all$diagnosis <- as.factor(data_all$diagnosis)
data_all$diagnosis <- as.integer(data_all$diagnosis)-1


### Summary of dataset in package
summary(data_all)
nrow(data_all)


### Correlation plot for whole dataset 
#pairs(data_all[-1])
findCorrelation(cor(data_all[-1]), cutoff = 0.75, names = TRUE)


### Splitting dataset dividing data 75/25 split
set.seed(99)
split <- sample.split(data_all$diagnosis, SplitRatio = 0.75)
head(split)

train_all <- subset(data_all, split == "TRUE")
test_all <- subset(data_all, split == "FALSE")


### Training model full model and summary output
logistic_full <- glm(diagnosis ~ ., data=train_all, family="binomial")
logistic_full

summary(logistic_full)

#Assessing Model Fit
#We can compute McFadden's R2 for our model using the pR2 function from the pscl package.

pR2(logistic_full)["McFadden"]
#A value of 0.9084415 is quite high for McFadden's R2, 
#which indicates that our model fits the data very well and has high predictive power.

#Variable Importance
varImp(logistic_full, sort = TRUE)

#calculate VIF values for each predictor variable in our model
vif(logistic_full)

#we can assume that multicollinearity is an issue in our model. So, we have 
#values above 5 indicate severe multicollinearity such that radius_worst and perimeter_worst.
# Set a VIF threshold. All the variables having higher VIF than threshold
#are dropped from the model
threshold=4.99


### Sequentially drop the variable with the largest VIF until
# all variables have VIF less than threshold
logistic_all <- logistic_full
drop=TRUE

aftervif=data.frame()
while(drop==TRUE) {
  vmodel=vif(logistic_all)
  aftervif=rbind.fill(aftervif,as.data.frame(t(vmodel)))
  if(max(vmodel)>threshold) {
    logistic_all=update(logistic_all,as.formula(paste(".","~",".","-",names(which.max(vmodel))))) }
  else { drop=FALSE }}

#Model after removing correlated Variables
summary(logistic_all)
vif(logistic_all)


### How variables removed sequentially
t_aftervif= as.data.frame(t(aftervif))

# Final (uncorrelated) variables with their VIFs
print(as.data.frame(vmodel))


### Use the Model to Make Predictions on test data
# Predict test data, converting to 0 or 1 based on 0.5 cutoff value
predict_reg <- predict(logistic_all, test_all, type = "response")
predict_reg <- ifelse(predict_reg > 0.5, 1, 0)
predict_reg <- as.vector(predict_reg)


### Model Diagnostics
# Diagnostics plots
par(mfrow = c(2,2))
plot(logistic_all, which = 1:4, main = "All Cancer Data")

### ROC-AUC Curve
ROCPred <- prediction(predict_reg, test_all$diagnosis)
ROCPer <- performance(ROCPred, measure = "tpr",
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

### Plotting curve
par(mfrow = c(1,1))
plot(ROCPer, main = "ROC Curve for All Cancer Data")
abline(a = 0, b = 1)
auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)


### Evaluating model accuracy
predict_reg <- factor(ifelse(predict_reg > 0.5, 1, 0),
                      labels = c("B", "M"))
test_all$diagnosis <- factor(ifelse(test_all$diagnosis > 0.5, 1, 0),
                             labels = c("B", "M"))
all_confusion <- caret::confusionMatrix(test_all$diagnosis, predict_reg,
                                        mode = 'everything',
                                        positive = 'M')
all_r2 <- pR2(logistic_all)["McFadden"]









###load dataset
data_mean <- read.csv("breast_cancer_mean.csv")

#r Setting diagnosis to factor variable: factor set 1 == "M", 0 == "B"
data_mean$diagnosis <- as.factor(data_mean$diagnosis)
data_mean$diagnosis <- as.integer(data_mean$diagnosis)-1


### Summary of dataset in package
summary(data_mean)
nrow(data_mean)


### Correlation plot for whole dataset 
#pairs(data_mean[-1])
findCorrelation(cor(data_mean[-1]), cutoff = 0.7, names = TRUE)


### Splitting dataset dividing data 75/25 split
set.seed(99)
split <- sample.split(data_mean$diagnosis, SplitRatio = 0.75)
head(split)

train_mean <- subset(data_mean, split == "TRUE")
test_mean <- subset(data_mean, split == "FALSE")


### Training model full model and summary output
logistic_mean <- glm(diagnosis ~ ., data=train_mean, family="binomial")
logistic_mean

summary(logistic_mean)

#Assessing Model Fit
#We can compute McFadden's R2 for our model using the pR2 function from the pscl package.

pR2(logistic_mean)["McFadden"]
#A value of 0.9084415 is quite high for McFadden's R2, 
#which indicates that our model fits the data very well and has high predictive power.

#Variable Importance
varImp(logistic_mean, sort = TRUE)

#calculate VIF values for each predictor variable in our model
vif(logistic_mean)

#we can assume that multicollinearity is an issue in our model. So, we have 
#values above 5 indicate severe multicollinearity such that radius_worst and perimeter_worst.
# Set a VIF threshold. All the variables having higher VIF than threshold
#are dropped from the model
threshold=4.99


### Sequentially drop the variable with the largest VIF until
# all variables have VIF less than threshold
drop=TRUE

aftervif=data.frame()
while(drop==TRUE) {
  vmodel=vif(logistic_mean)
  aftervif=rbind.fill(aftervif,as.data.frame(t(vmodel)))
  if(max(vmodel)>threshold) {
    logistic_mean=update(logistic_mean,as.formula(paste(".","~",".","-",names(which.max(vmodel))))) }
  else { drop=FALSE }}

#Model after removing correlated Variables
summary(logistic_mean)
vif(logistic_mean)


### How variables removed sequentially
t_aftervif= as.data.frame(t(aftervif))

# Final (uncorrelated) variables with their VIFs
print(as.data.frame(vmodel))


### Use the Model to Make Predictions on test data
# Predict test data, converting to 0 or 1 based on 0.5 cutoff value
predict_reg <- predict(logistic_mean, test_mean, type = "response")
predict_reg <- ifelse(predict_reg > 0.5, 1, 0)
predict_reg <- as.vector(predict_reg)


### Model Diagnostics
# Diagnostic plots
par(mfrow = c(2,2))
plot(logistic_mean, which = 1:4, main = "Mean Cancer Data")

# ROC-AUC Curve
ROCPred <- prediction(predict_reg, test_mean$diagnosis)
ROCPer <- performance(ROCPred, measure = "tpr",
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

### Plotting curve
par(mfrow = c(1,1))
plot(ROCPer, main = "ROC Curve for Mean Cancer Data")
abline(a = 0, b = 1)
auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)


### Evaluating model accuracy
predict_reg <- factor(ifelse(predict_reg > 0.5, 1, 0),
                      labels = c("B", "M"))
test_mean$diagnosis <- factor(ifelse(test_mean$diagnosis > 0.5, 1, 0),
                             labels = c("B", "M"))
mean_confusion<- caret::confusionMatrix(test_mean$diagnosis, predict_reg,
                       mode = 'everything',
                       positive = 'M')
mean_r2 <- pR2(logistic_mean)["McFadden"]

















###load dataset
data_worst <- read.csv("breast_cancer_worst.csv")

#r Setting diagnosis to factor variable: factor set 1 == "M", 0 == "B"
data_worst$diagnosis <- as.factor(data_worst$diagnosis)
data_worst$diagnosis <- as.integer(data_worst$diagnosis)-1


### Summary of dataset in package
summary(data_worst)
nrow(data_worst)


### Correlation plot for whole dataset 
#pairs(data_worst[-1])
findCorrelation(cor(data_worst[-1]), cutoff = 0.7, names = TRUE)


### Splitting dataset dividing data 75/25 split
set.seed(99)
split <- sample.split(data_worst$diagnosis, SplitRatio = 0.75)
head(split)

train_worst <- subset(data_worst, split == "TRUE")
test_worst <- subset(data_worst, split == "FALSE")


### Training model full model and summary output
logistic_worst <- glm(diagnosis ~ ., data=train_worst, family="binomial")
logistic_worst

summary(logistic_worst)

#Assessing Model Fit
#We can compute McFadden's R2 for our model using the pR2 function from the pscl package.

pR2(logistic_worst)["McFadden"]
#A value of 0.9084415 is quite high for McFadden's R2, 
#which indicates that our model fits the data very well and has high predictive power.

#Variable Importance
varImp(logistic_worst, sort = TRUE)

#calculate VIF values for each predictor variable in our model
vif(logistic_worst)

#we can assume that multicollinearity is an issue in our model. So, we have 
#values above 5 indicate severe multicollinearity such that radius_worst and perimeter_worst.
# Set a VIF threshold. All the variables having higher VIF than threshold
#are dropped from the model
threshold=4.99


### Sequentially drop the variable with the largest VIF until
# all variables have VIF less than threshold
drop=TRUE

aftervif=data.frame()
while(drop==TRUE) {
  vmodel=vif(logistic_worst)
  aftervif=rbind.fill(aftervif,as.data.frame(t(vmodel)))
  if(max(vmodel)>threshold) {
    logistic_worst=update(logistic_worst,as.formula(paste(".","~",".","-",names(which.max(vmodel))))) }
  else { drop=FALSE }}

#Model after removing correlated Variables
summary(logistic_worst)
vif(logistic_worst)


### How variables removed sequentially
t_aftervif= as.data.frame(t(aftervif))

# Final (uncorrelated) variables with their VIFs
print(as.data.frame(vmodel))


### Use the Model to Make Predictions on test data
# Predict test data, converting to 0 or 1 based on 0.5 cutoff value
predict_reg <- predict(logistic_worst, test_worst, type = "response")
predict_reg <- ifelse(predict_reg > 0.5, 1, 0)
predict_reg <- as.vector(predict_reg)


### Model Diagnostics
# Diagnostic plots
par(mfrow = c(2,2))
plot(logistic_worst, which = 1:4, main = "Worst Cancer Data")

# ROC-AUC Curve
ROCPred <- prediction(predict_reg, test_worst$diagnosis)
ROCPer <- performance(ROCPred, measure = "tpr",
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

### Plotting curve
par(mfrow = c(1,1))
plot(ROCPer, main = "ROC Curve for Worst Cancer Data")
abline(a = 0, b = 1)
auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)


### Evaluating model accuracy
predict_reg <- factor(ifelse(predict_reg > 0.5, 1, 0),
                      labels = c("B", "M"))
test_worst$diagnosis <- factor(ifelse(test_worst$diagnosis > 0.5, 1, 0),
                              labels = c("B", "M"))
worst_confusion<- caret::confusionMatrix(test_worst$diagnosis, predict_reg,
                       mode = 'everything',
                       positive = 'M')
worst_r2 <- pR2(logistic_worst)["McFadden"]







### Total Model Comparisons
all_confusion
all_r2

mean_confusion
mean_r2

worst_confusion
worst_r2
