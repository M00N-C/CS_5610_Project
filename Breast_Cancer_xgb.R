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
library(pROC)

start.time <- Sys.time()
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


## Hyper-parameter tuning using a random search

### parameters: max_depth, eta, subsample, colsample_bytree, and min_child_weight
all_low_err_list <- list()
all_parameters_list <- list()
set.seed(99)
for(i in 1:3000){
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 max_depth = sample(3:25, 1),
                 eta = runif(1, 0.01, 0.3),
                 subsample = runif(1, 0.5, 1),
                 colsample_bytree = runif(1, 0.5, 1),
                 min_child_weight = sample(0:10, 1)
                )
  
  parameters <- as.data.frame(params)
  all_parameters_list[[i]] <- parameters
}

all_parameters_df <- do.call(rbind, all_parameters_list) #df containing random search params

### Fitting xgboost models based on search parameters
for (row in 1:nrow(all_parameters_df)){
  set.seed(99)
  all_tmp_mdl <- xgb.cv(data = all_dtrain,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       nfold = 5,
                       prediction = TRUE,
                       max_depth = all_parameters_df$max_depth[row],
                       eta = all_parameters_df$eta[row],
                       subsample = all_parameters_df$subsample[row],
                       colsample_bytree = all_parameters_df$colsample_bytree[row],
                       min_child_weight = all_parameters_df$min_child_weight[row],
                       nrounds = 200,
                       eval_metric = "error",
                       early_stopping_rounds = 20,
                       print_every_n = 500,
                       verbose = 0
                    )
  
  #this is the lowest error for the iteration
  all_low_err <- as.data.frame(1 - min(all_tmp_mdl$evaluation_log$test_error_mean))
  all_low_err_list[[row]] <- all_low_err
}

all_low_err_df <- do.call(rbind, all_low_err_list) #accuracies 
all_randsearch <- cbind(all_low_err_df, all_parameters_df) #data frame with everything

###Reformatting the dataframe
all_randsearch <- all_randsearch %>%
  dplyr::rename(val_acc = '1 - min(all_tmp_mdl$evaluation_log$test_error_mean)') %>%
  dplyr::arrange(-val_acc)

###Grabbing just the top model
all_randsearch_best <- all_randsearch[1,]

###Storing best parameters in list
all_best_params <- list(booster = all_randsearch_best$booster,
                        objective = all_randsearch_best$objective,
                        max_depth = all_randsearch_best$max_depth,
                        eta = all_randsearch_best$eta,
                        subsample = all_randsearch_best$subsample,
                        colsample_bytree = all_randsearch_best$colsample_bytree,
                        min_child_weight = all_randsearch_best$min_child_weight)

### Finding the best nround parameter for the model using 5-fold cross validation
set.seed(99)
all_xgbcv <- xgb.cv(params = all_best_params,
                    data = all_dtrain,
                    nrounds = 500,
                    nfold = 5,
                    prediction = TRUE,
                    print_every_n = 50,
                    early_stopping_rounds = 25,
                    eval_metric = "error",
                    verbose = 0
                    )
all_xgbcv$best_iteration

## Final model
set.seed(99)
all_best_xgb <- xgb.train(params = all_best_params,
                          data = all_dtrain,
                          nrounds = all_xgbcv$best_iteration,
                          eval_metric = "error",
                          )

xgb.save(all_best_xgb, 'final_xgb_cancerall')

cancer_all.pred <- predict(all_best_xgb, all_dtest)
cancer_all.pred <- factor(ifelse(cancer_all.pred > 0.5, 1, 0),
                          labels = c("B", "M"))
confusionMatrix(cancer_all.pred, test_all$diagnosis,
                mode = 'everything',
                positive = 'M')

## Visualizations
all_impt_mtx <- xgb.importance(feature_names = colnames(test_all_data), model = all_best_xgb)
xgb.plot.importance(importance_matrix = all_impt_mtx,
                      xlab = "Variable Importance")


### ROC curve for 5-fold CV random parameter search
all_randsearch_roc <- roc(response = train_all_label,
                          predictor = all_tmp_mdl$pred,
                          print.auc = TRUE,
                          plot = TRUE)

### ROC curve for 5-fold CV nround parameter search
all_nround_roc <- roc(response = train_all_label,
                          predictor = all_xgbcv$pred,
                          print.auc = TRUE,
                          plot = TRUE)







# Gradient boosting cancer_mean data
## Subsetting data into training and test data
set.seed(99)
sampl_mean <- sample.split(cancer_mean$diagnosis, SplitRatio = 0.75)
train_mean <- subset(cancer_mean, sampl_mean == TRUE)
test_mean  <- subset(cancer_mean, sampl_mean != TRUE)

## Creating the independent variable and label matricies of train/test data
train_mean_data  <- as.matrix(train_mean[-1])
train_mean_label <- train_mean$diagnosis
## Converting labels to 0,1 where "M" is coded at 1
train_mean_label <- as.integer(train_mean_label)-1
train_mean$diagnosis[1:5]; train_mean_label[1:5]
## Repeat for test dataset
test_mean_data   <- as.matrix(test_mean[-1])
test_mean_label  <- test_mean$diagnosis
test_mean_label  <- as.integer(test_mean_label)-1
test_mean$diagnosis[1:5]; test_mean_label[1:5]

## Formatting data for XGBoost matricies
mean_dtrain <- xgb.DMatrix(data = train_mean_data, label = train_mean_label)
mean_dtest  <- xgb.DMatrix(data = test_mean_data, label = test_mean_label)


## Hyper-parameter tuning using a random search

### parameters: max_depth, eta, subsample, colsample_bytree, and min_child_weight
mean_low_err_list <- list()
mean_parameters_list <- list()
set.seed(99)
for(i in 1:3000){
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 max_depth = sample(3:25, 1),
                 eta = runif(1, 0.01, 0.3),
                 subsample = runif(1, 0.5, 1),
                 colsample_bytree = runif(1, 0.5, 1),
                 min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(params)
  mean_parameters_list[[i]] <- parameters
}
mean_parameters_df <- do.call(rbind, mean_parameters_list) #df containing random search params

### Fitting xgboost models based on search parameters
for (row in 1:nrow(mean_parameters_df)){
  set.seed(99)
  mean_tmp_mdl <- xgb.cv(data = mean_dtrain,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       nfold = 5,
                       prediction = TRUE,
                       max_depth = mean_parameters_df$max_depth[row],
                       eta = mean_parameters_df$eta[row],
                       subsample = mean_parameters_df$subsample[row],
                       colsample_bytree = mean_parameters_df$colsample_bytree[row],
                       min_child_weight = mean_parameters_df$min_child_weight[row],
                       nrounds = 200,
                       eval_metric = "error",
                       early_stopping_rounds = 20,
                       print_every_n = 500,
                       verbose = 0
                      ) 
  
  #this is the lowest error for the iteration
  mean_low_err <- as.data.frame(1 - min(mean_tmp_mdl$evaluation_log$test_error_mean))
  mean_low_err_list[[row]] <- mean_low_err
}

mean_low_err_df <- do.call(rbind, mean_low_err_list) #accuracies 
mean_randsearch <- cbind(mean_low_err_df, mean_parameters_df) #data frame with everything

###Reformatting the dataframe
mean_randsearch <- mean_randsearch %>%
  dplyr::rename(val_acc = '1 - min(mean_tmp_mdl$evaluation_log$test_error_mean)') %>%
  dplyr::arrange(-val_acc)

###Grabbing just the top model
mean_randsearch_best <- mean_randsearch[1,]

### Storing best parameters in list
mean_best_params <- list(booster = mean_randsearch_best$booster,
                         objective = mean_randsearch_best$objective,
                         max_depth = mean_randsearch_best$max_depth,
                         eta = mean_randsearch_best$eta,
                         subsample = mean_randsearch_best$subsample,
                         colsample_bytree = mean_randsearch_best$colsample_bytree,
                         min_child_weight = mean_randsearch_best$min_child_weight)

### Finding the best nround parameter for the model using 5-fold cross validation
set.seed(99)
mean_xgbcv <- xgb.cv(params = mean_best_params,
                      data = mean_dtrain,
                      nrounds = 500,
                      nfold = 5,
                      prediction = TRUE,
                      print_every_n = 50,
                      early_stopping_rounds = 25,
                      eval_metric = "error",
                      verbose = 0
                      )
mean_xgbcv$best_iteration


## Final model
set.seed(99)
mean_best_xgb <- xgb.train(params = mean_best_params,
                          data = mean_dtrain,
                          nrounds = mean_xgbcv$best_iteration,
                          eval_metric = "error",
                          )

xgb.save(mean_best_xgb, 'final_xgb_cancermean')

cancer_mean.pred <- predict(mean_best_xgb, mean_dtest)
cancer_mean.pred <- factor(ifelse(cancer_mean.pred > 0.5, 1, 0),
                          labels = c("B", "M"))
confusionMatrix(cancer_mean.pred, test_mean$diagnosis,
                mode = 'everything',
                positive = 'M')


## Visualizations
mean_impt_mtx <- xgb.importance(feature_names = colnames(test_mean_data), model = mean_best_xgb)
xgb.plot.importance(importance_matrix = mean_impt_mtx,
                    xlab = "Variable Importance")

### ROC curve for 5-fold CV random parameter search
mean_randsearch_roc <- roc(response = train_mean_label,
                            predictor = mean_tmp_mdl$pred,
                            print.auc = TRUE,
                            plot = TRUE)

### ROC curve for 5-fold CV nround parameter search
mean_nround_roc <- roc(response = train_mean_label,
                            predictor = mean_xgbcv$pred,
                            print.auc = TRUE,
                            plot = TRUE)








# Gradient boosting cancer_worst data
## Subsetting data into training and test data
set.seed(99)
sampl_worst <- sample.split(cancer_worst$diagnosis, SplitRatio = 0.75)
train_worst <- subset(cancer_worst, sampl_worst == TRUE)
test_worst  <- subset(cancer_worst, sampl_worst != TRUE)

## Creating the independent variable and label matricies of train/test data
train_worst_data  <- as.matrix(train_worst[-1])
train_worst_label <- train_worst$diagnosis
## Converting labels to 0,1 where "M" is coded at 1
train_worst_label <- as.integer(train_worst_label)-1
train_worst$diagnosis[1:5]; train_worst_label[1:5]
## Repeat for test dataset
test_worst_data   <- as.matrix(test_worst[-1])
test_worst_label  <- test_worst$diagnosis
test_worst_label  <- as.integer(test_worst_label)-1
test_worst$diagnosis[1:5]; test_worst_label[1:5]

## Formatting data for XGBoost matricies
worst_dtrain <- xgb.DMatrix(data = train_worst_data, label = train_worst_label)
worst_dtest  <- xgb.DMatrix(data = test_worst_data, label = test_worst_label)


## Hyper-parameter tuning using a random search

### parameters: max_depth, eta, subsample, colsample_bytree, and min_child_weight
worst_low_err_list <- list()
worst_parameters_list <- list()
set.seed(99)
for(i in 1:3000){
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 max_depth = sample(3:25, 1),
                 eta = runif(1, 0.01, 0.3),
                 subsample = runif(1, 0.5, 1),
                 colsample_bytree = runif(1, 0.5, 1),
                 min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(params)
  worst_parameters_list[[i]] <- parameters
}
worst_parameters_df <- do.call(rbind, worst_parameters_list) #df containing random search params

### Fitting 5-fold CV xgboost models based on search parameters 
for (row in 1:nrow(worst_parameters_df)){
  set.seed(99)
  worst_tmp_mdl <- xgb.cv(data = worst_dtrain,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       nfold = 5,
                       prediction = TRUE,
                       max_depth = worst_parameters_df$max_depth[row],
                       eta = worst_parameters_df$eta[row],
                       subsample = worst_parameters_df$subsample[row],
                       colsample_bytree = worst_parameters_df$colsample_bytree[row],
                       min_child_weight = worst_parameters_df$min_child_weight[row],
                       nrounds = 200,
                       eval_metric = "error",
                       early_stopping_rounds = 20,
                       print_every_n = 500,
                       verbose = 0
                       )
                       
  
  #this is the lowest error for the iteration
  worst_low_err <- as.data.frame(1 - min(worst_tmp_mdl$evaluation_log$test_error_mean))
  worst_low_err_list[[row]] <- worst_low_err
}

worst_low_err_df <- do.call(rbind, worst_low_err_list) #accuracies 
worst_randsearch <- cbind(worst_low_err_df, worst_parameters_df) #data frame with everything

###Reformatting the dataframe
worst_randsearch <- worst_randsearch %>%
  dplyr::rename(val_acc = '1 - min(worst_tmp_mdl$evaluation_log$test_error_mean)') %>%
  dplyr::arrange(-val_acc)

###Grabbing just the top model
worst_randsearch_best <- worst_randsearch[1,]

### Storing best parameters in list
worst_best_params <- list(booster = worst_randsearch_best$booster,
                          objective = worst_randsearch_best$objective,
                          max_depth = worst_randsearch_best$max_depth,
                          eta = worst_randsearch_best$eta,
                          subsample = worst_randsearch_best$subsample,
                          colsample_bytree = worst_randsearch_best$colsample_bytree,
                          min_child_weight = worst_randsearch_best$min_child_weight)

### Finding the best nround parameter for the model using 5-fold cross validation
set.seed(99)
worst_xgbcv <- xgb.cv(params = worst_best_params,
                    data = worst_dtrain,
                    nrounds = 500,
                    nfold = 5,
                    prediction = TRUE, 
                    print_every_n = 50,
                    early_stopping_rounds = 25,
                    eval_metric = "error",
                    verbose = 0
                    )
worst_xgbcv$best_iteration


## Final model
set.seed(99)
worst_best_xgb <- xgb.train(params = worst_best_params,
                            data = worst_dtrain,
                            nrounds = worst_xgbcv$best_iteration,
                            eval_metric = "error"
                            )
xgb.save(worst_best_xgb, 'final_xgb_cancerworst')

cancer_worst.pred <- predict(worst_best_xgb, worst_dtest)
cancer_worst.pred <- factor(ifelse(cancer_worst.pred> 0.5, 1, 0),
                          labels = c("B", "M"))
confusionMatrix(cancer_worst.pred, test_worst$diagnosis,
                mode = 'everything',
                positive = 'M')


## Visualizations

### variable importance plot
worst_impt_mtx <- xgb.importance(feature_names = colnames(test_worst_data), model = worst_best_xgb)
xgb.plot.importance(importance_matrix = worst_impt_mtx,
                    xlab = "Variable Importance")

### ROC curve for 5-fold CV random parameter search
worst_randsearch_roc <- roc(response = train_worst_label,
                           predictor = worst_tmp_mdl$pred,
                           print.auc = TRUE,
                           plot = TRUE)

### ROC curve for 5-fold CV nround parameter search
worst_nround_roc <- roc(response = train_worst_label,
                           predictor = worst_xgbcv$pred,
                           print.auc = TRUE,
                           plot = TRUE)


stop.time <- Sys.time()
time.elapsed <- stop.time - start.time
time.elapsed

confusionMatrix(cancer_all.pred, test_all$diagnosis,
                mode = 'everything',
                positive = 'M')
confusionMatrix(cancer_mean.pred, test_mean$diagnosis,
                mode = 'everything',
                positive = 'M')
confusionMatrix(cancer_worst.pred, test_worst$diagnosis,
                mode = 'everything',
                positive = 'M')
