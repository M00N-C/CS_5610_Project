## CS 5610 Project Exploratory Data Analysis ##
library(tidyverse)
#install.packages('GGally')
library(GGally)
#Loading dataset
breast_cancer <- readxl::read_xlsx('Breast Cancer data - CS 5610.xlsx')

#Removing id and standard error columns
b_cancer <- breast_cancer[, -1]
b_cancer <- b_cancer[, !grepl('_se', colnames(b_cancer))]
colnames(b_cancer)
colnames(b_cancer)[c(9, 19)] <- c("concave_points_mean", "concave_points_worst")
str(b_cancer)

#converting diagnosis to factor
b_cancer$diagnosis <- b_cancer$diagnosis %>% as.factor()

#Overview of data
summary(b_cancer)
ggcorr(b_cancer[, -1], method = c('everything', 'pearson'), 
       label = TRUE,
       label_round = 2,
       label_size = 3)

#Plot histograms of covariates
ggplot(gather(b_cancer[,-1]), aes(x = value, color = key, fill = key)) +
  geom_histogram(bins = 20) +
  facet_wrap(~key, scales = 'free_x')

#Plotting histograms of covariates grouped by diagnosis, for mean/worst
hist <- list()
for(i in names(b_cancer[,-1])){
  hist[[i]] <- ggplot(data = b_cancer, aes_string(x = i,
                   fill = "diagnosis")) +
                   geom_histogram(position = 'identity')     
}

#Worst count covariates
grep('worst', names(b_cancer[,-1]))
gridExtra::grid.arrange(hist[[11]], hist[[12]], hist[[13]], hist[[14]], hist[[15]],
                        hist[[16]], hist[[17]], hist[[18]], hist[[19]], hist[[20]],
                        nrow = 4)
#Mean count covariates
grep('mean', names(b_cancer[,-1]))
gridExtra::grid.arrange(hist[[1]], hist[[2]], hist[[3]], hist[[4]], hist[[5]],
                        hist[[6]], hist[[7]], hist[[8]], hist[[9]], hist[[10]],
                        nrow = 4)


#Partitioning dataset into mean/worst covariates
cancer_mean <- b_cancer[, c(1, grep('mean', names(b_cancer)))]
colnames(cancer_mean)

cancer_worst <- b_cancer[, c(1, grep('worst', names(b_cancer)))]
colnames(cancer_worst)
write.csv(cancer_mean, "breast_cancer_mean.csv", row.names = FALSE)
write.csv(cancer_worst, "breast_cancer_worst.csv", row.names = FALSE)
