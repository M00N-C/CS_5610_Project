# Gradient Boosting models (XGBoost) #
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

#Loading packages
#install.packages('xgboost')
library(xgboost)
library(caret)
library(caTools)
library(tidyverse)


# Gradient boosting cancer_all data

## Subsetting data into training and test data
set.seed(99)
sampl_all <- sample.split(cancer_all$diagnosis, SplitRatio = 0.75)
train_all <- subset(cancer_all, sampl_all == TRUE)
test_all  <- subset(cancer_all, sampl_all != TRUE)

## Creating the independent variable and label matricies of train/test data
train_all_data  <- as.matrix(train_all[-1])
train_all_label <- train_all$diagnosis
## Converting labels to 0,1 where "M" is coded at 1
train_all_label <- as.integer(train_all_label)-1
train_all$diagnosis[1:5]; train_all_label[1:5]
## Repeat for test dataset
test_all_data   <- as.matrix(test_all[-1])
test_all_label  <- test_all$diagnosis
test_all_label  <- as.integer(test_all_label)-1
test_all$diagnosis[1:5]; test_all_label[1:5]

## Formatting data for XGBoost matricies
all_dtrain <- xgb.DMatrix(data = train_all_data, label = train_all_label)
all_dtest  <- xgb.DMatrix(data = test_all_data, label = test_all_label)
