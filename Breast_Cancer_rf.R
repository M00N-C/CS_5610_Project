# Random Forest COde #
#Loading datasets
breast_cancer <- readxl::read_xlsx('Breast Cancer data - CS 5610.xlsx')
cancer_mean   <- read.csv('breast_cancer_mean.csv')
cancer_worst  <- read.csv('breast_cancer_worst.csv')
#Removing SE columns and renaming two columns for cancer_all
cancer_all    <- breast_cancer[, -1]
cancer_all    <- cancer_all[, !grepl('_se', colnames(cancer_all))]
colnames(cancer_all)[c(9, 19)] <- c("concave_points_mean", "concave_points_worst")
#Setting 'diagnosis' to factor variable
cancer_all$diagnosis   <- as.factor(cancer_all$diagnosis)
cancer_mean$diagnosis  <- as.factor(cancer_mean$diagnosis)
cancer_worst$diagnosis <- as.factor(cancer_worst$diagnosis)

#Loading required libraries
library(tidyverse)
library(randomForest)
library(caTools)
library(caret)

#RF for cancer_all data
set.seed(99)
sampl_all <- sample.split(cancer_all$diagnosis, SplitRatio = 0.75)
train_all <- subset(cancer_all, sampl_all == TRUE)
test_all  <- subset(cancer_all, sampl_all != TRUE)

## hyperparameter tuning
control <- trainControl(method = 'cv', number = 5, search = 'grid')
tunegrid <- expand.grid(mtry = c(1:ncol(train_all)))
set.seed(99)
test_rf <- train(diagnosis ~., data = train_all, method = 'rf',
                 metric = 'Accuracy',
                 tuneGrid = tunegrid,
                 trControl = control)
plot(test_rf, main = "CV Accuracy per Number of Included Predictors",
     sub = "All Cancer Data")
test_rf$bestTune

## building training model
set.seed(99)
rf_all <- randomForest(diagnosis ~., data = train_all, 
                       ntree = 500,
                       mtry = 20,
                       importance = TRUE)

## testing model, confusion matrix, plots
cancer_all.pred <- predict(rf_all, newdata = test_all)
confusionMatrix(cancer_all.pred, test_all$diagnosis,
                mode = 'everything',
                positive = 'M')
varImpPlot(rf_all, main = "Variable Importance: All Cancer Data")










#RF for cancer_mean data
set.seed(99)
sampl_mean <- sample.split(cancer_mean$diagnosis, SplitRatio = 0.75)
train_mean <- subset(cancer_mean, sampl_mean == TRUE)
test_mean  <- subset(cancer_mean, sampl_mean != TRUE)

## hyperparameter tuning
control <- trainControl(method = 'cv', number = 5, search = 'grid')
tunegrid <- expand.grid(mtry = c(1:ncol(train_mean)))
set.seed(99)
test_rf <- train(diagnosis ~., data = train_mean, method = 'rf',
                 metric = 'Accuracy',
                 tuneGrid = tunegrid,
                 trControl = control)
plot(test_rf, main = "CV Accuracy per Number of Included Predictors",
     sub = "Mean Cancer Data")
test_rf$bestTune

## building training model
set.seed(99)
rf_mean <- randomForest(diagnosis ~., data = train_mean, 
                       ntree = 500,
                       mtry = 8,
                       importance = TRUE)

## testing model, confusion matrix, plots
cancer_mean.pred <- predict(rf_mean, newdata = test_mean)
confusionMatrix(cancer_mean.pred, test_mean$diagnosis,
                mode = 'everything',
                positive = 'M')
varImpPlot(rf_mean, main = "Variable Importance: Mean Cancer Data")












#RF for cancer_worst data
set.seed(99)
sampl_worst <- sample.split(cancer_worst$diagnosis, SplitRatio = 0.75)
train_worst <- subset(cancer_worst, sampl_worst == TRUE)
test_worst  <- subset(cancer_worst, sampl_worst != TRUE)

## hyperparameter tuning
control <- trainControl(method = 'cv', number = 5, search = 'grid')
tunegrid <- expand.grid(mtry = c(1:ncol(train_worst)))
set.seed(99)
test_rf <- train(diagnosis ~., data = train_worst, method = 'rf',
                 metric = 'Accuracy',
                 tuneGrid = tunegrid,
                 trControl = control)
plot(test_rf, main = "CV Accuracy per Number of Included Predictors",
     sub = "Worst Cancer Data")
test_rf$bestTune

## building training model
set.seed(99)
rf_worst <- randomForest(diagnosis ~., data = train_worst,
                         ntree = 500, 
                         mtry = 10,
                         importance = TRUE)

## testing model, confusion matrix, plots
cancer_worst.pred <- predict(rf_worst, newdata = test_worst)
confusionMatrix(cancer_worst.pred, test_worst$diagnosis,
                mode = 'everything',
                positive = 'M')

varImpPlot(rf_worst, main = "Variable Importance: Worst Cancer Data")

