#Rohan Bapat (rb2te)
#Caitlin Dreisbach (cnd2y) 
#Yi Hao (yh8a)
#SYS 6018: Kaggle data mining competition

#Packages
library("tidyverse") #data wrangling and manipulation
install.packages("RWeka") #collection of machine learning functions for KNN
library("RWeka")
install.packages("rminer") #for data mining classification and regression methods
library("rminer")
install.packages("RCurl")
library("RCurl") #essential for inclusion of URL
library("psych") #best for descriptive statistics
install.packages("randomForest") #will need this for the random forrest modeling
library("randomForest") #calls randomForest package for random forrest model
install.packages("rJava") #essential for RWeka to function
library("rJava")

#---------------------- Read in data
# Read traindataset
trainURL <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-house-prices/master/train.csv') 
train_data <- read.csv(text = trainURL)

#---------------------- General Descriptive Statistics and Exploratory Plotting of Training Data
#Descriptive statistics
summary(train_data$SalePrice) #mean = $180,921.20 max = $755,000 min = $34,900

#Are there trends in sale price by year? ANOVA comparison on sale prices by year
describeBy(train_data$SalePrice, train_data$YrSold) #describes descriptive stats by year sold
fit <- aov(SalePrice ~YrSold, data=train_data) #creates an ANOVA comparison for year sold on sale price
summary(fit) #provides and ANOVA sumamry table
#A non-signficant result (p = 0.269) shows that year is not a significant variable

#Is sale condition an important variable? 
fit2 <- aov(SalePrice ~ SaleCondition, data=train_data) #ANOVA comparison function
summary(fit2) #creates an ANOVA summary table that shoes sale condition is a
#statistically significant variable.
#               Df    Sum Sq   Mean Sq F value Pr(>F)    
#SaleCondition    5 1.248e+12 2.495e+11   45.58 <2e-16 ***
#Residuals     1454 7.960e+12 5.475e+09  

#graph and visualize variables
hist(train_data$SalePrice/1000, breaks=50) #creates a histogram for sale price
plot(train_data$SalePrice,pch=20,cex=.2) #plots overall sale price, can see grouping horizontally at the bottom

# ---------------------- Data Cleaning Functions

#Refer notes from author of dataset - https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
#Author of dataset recommends removing houses more than 4000 square feet
train_data <- train_data[train_data['GrLivArea']<4000,]

#Count the NAs across all the columns in the training data, sums the length of NA values
na_count <-sapply(train_data, function(y) sum(length(which(is.na(y))))); na_count

#Subsets and removes columns which missings that are likely not significant in the modeling procedures
train_data2 <- subset(train_data,select = -c(Alley,FireplaceQu,PoolQC, #removes alley, fireplace, pool quality
                                             Fence,MiscFeature,RoofMatl, #fence, miscellaneous features, roof material
                                             Exterior1st, Condition2, Exterior2nd)) #exterior qualities and condition

# ---------------------- Build Functions to Impute Missing Value Data

#Build function to impute median value for missing numerical variables
impute_median <- function(df, colnm){ #assigns imputation to new variable impute_median
  for(cols in colnm){ #for every column in all the columns
    print(cols) #print the column
    df[is.na(df[cols]),cols] <- median(df[,cols], na.rm = T) #if the columns include missing values take the median
  }
  return(df) #return the new dataset
}
#Impute test levels not found in train levels
impute_level <- function(df_train, df_test, colnm){ #create a function to assess levels in training and testing data
  for(col in colnm){ #for every column in all the columns
    train_levels_table <- table(df_train[,col]) #create a table of the columns in the training set
    max_train_level <- names(which.max(train_levels_table)) #take the maximum number from the table and provide the names
    train_levels <- levels(df_train[,col]) #identify the levels of all the columns in the training set
    test_levels <- levels(df_test[,col]) #identify the levels of all the columns in the test set
    msg_levels <- test_levels[!test_levels %in% train_levels] #identify the levels that are not in testing that are in training
    df_test[df_test[,col] %in% msg_levels,col] <- max_train_level #identify the columns in the combined levels variable
    df_test[,col] <- droplevels(df_test[,col]) #drop the levels of the test that are unique and reassign to the testing dataframe
  }
  return(df_test) #return a testing dataset 
}

# Build function to impute "none" for categorical variables
impute_none <- function(df, colnm){ #
  for(cols in colnm){ #for every column in all the columns
    levels(df[,cols]) <- c(levels(df[,cols]),"none") #identify the column levels
    df[is.na(df[cols]),cols]<- "none" #assigns none to categorical variables that have missing variables
  }
  
  return(df) #return the new dataframe with imputed variables
}

# ---------------------- Utilize Functions to Impute Missing Value Data

#Impute missings in numerical with median values
train_data2 <- impute_median(train_data2,c("LotFrontage","MasVnrArea"))

#list of factors in train data
char_cols <- sapply(train_data2, is.factor) #create a new variable that applies factor to all character columns

# Impute factor variables with "none"
train_data2 <- impute_none(train_data2, colnames(train_data2[,char_cols]) ) 

# Convert numerical variables which are actually categorical
num_cat_cols <- c('MSSubClass', 'BldgType') #combine two variables into new object
train_data2[,'MSSubClass'] <- factor(train_data2[,'MSSubClass']) #factor the MSSubClass column
train_data2[,'BldgType'] <- factor(train_data2[,'BldgType']) #factor the BldgType column

# Transform SalePrice to logarithmic
#This is important to transform the overall data to a normalized set to reduce bias
#related to the inital skewness as seen by the exploratory data above.
#This apply function creates a normal distribution of the data by taking the log.
train_data2$SalePrice <- sapply(train_data2$SalePrice, log)

# ---------------------- Cross Validation

# Cross validation
train_test_split <- 0.7

# Create randomly generated sample for cross validation
cv_sample <- sample(1:nrow(train_data2),round(train_test_split*nrow(train_data2)))

# Create test and train datasets
train1 <- train_data2[cv_sample,] #subsets data into training
test1 <- train_data2[-cv_sample,] #subsets data into testing

#write CSV for cross validating training and testing if desired
write.csv(train1, file = "Crossvalid_train_data_team1-4.csv", row.names = FALSE)
write.csv(test1, file = "Crossvalid_test_data_team1-4.csv", row.names = FALSE)

# ---------------------- Building a Linear Model to Assess Signficance and Relationship of the Variables

#Build linear model
lin_model <- lm(SalePrice~., train_data2) #creates a linear model for sale price across all variables in the subset training data
anova_lin_model <- anova(lin_model) #creates an ANOVA table to look at significance of the testing
anova_significance <- data.frame(anova_lin_model$`Pr(>F)`) #creates a dataframe of the p-values from the significance testing
anova_significance$varname <- rownames(anova_lin_model) #creates new variable from the rownames of the linear model
colnames(anova_significance) <- c("p_value", "varname") #combines columns p-value and varname
anova_significance <- anova_significance[order(anova_significance$p_value),] #order the p-values in the signifance object

#Iteratively test variables based on increasing p value
for(i in 3:nrow(anova_significance)){ #for each row in the signifiance object based on the ANOVA testing
  train_df <- train1[,c(head(anova_significance$varname,i),"SalePrice")] #list the top sale price in the signifance object in the training set
  test_df <- test1[,c(head(anova_significance$varname,i),"SalePrice")] #list the top sale price in the signifance object in the test set
  temp_lin_model <- lm(SalePrice~., train_df) #creates a temporary linear model object with the signifiance dataframe
  
  temp_pred <- predict(temp_lin_model,test1) #uses the temporary linear model to predict based on the testing data
  mse <- sum((test_df$SalePrice-temp_pred)^2) #computes the mean square error for all variables
  print(i) #prints list of variable numbers
  print(mse) #prints list of mean square errors
}

#Based on the above analysis, we find that the mean square error is minimized for the first 19 variables
#Hence, considering only the first 28 variables by p_value

# Build model with optimum number of variables
train_data3 <- train_data2[,c(head(anova_significance$varname,28),"SalePrice")] #uses the 28 variables and lists top in the training data
lin_model2 <- lm(SalePrice~., train_data3) #create a new linear model with sale price and the top signifiance variables

#---------------------- Building a Linear Model to Answer the Kaggle Problem

#Import test dataset
testURL <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-house-prices/master/test.csv') 
test_data <- read.csv(text = testURL)
test_data2 <- test_data[,c(head(anova_significance$varname, 28))] #utilizes the variables of signifance from testing

#Convert numerical variables which are actually categorical
num_cat_cols <- c('MSSubClass') 
test_data2[,num_cat_cols] <- sapply(test_data2[,num_cat_cols], factor) #applies factoring to the categorical columns

#Impute missing values with 
test_data2[is.na(test_data2['MSZoning']), 'MSZoning'] <- names(which.max(table(test_data['MSZoning'])))

#Clean test data
test_data2 <- impute_median(test_data2, c('LotFrontage', 'BsmtUnfSF','BsmtFinSF1','MasVnrArea','GarageCars'))
test_data2 <- impute_none(test_data2, c('BsmtQual','BsmtExposure','BsmtFinType1','MasVnrType'))

#Ensure test data variables have same levels as train data
for(i in 1:ncol(test_data2)){ #for all values in the test data
  levels(test_data2[,i]) <- levels(train_data3[,i]) #confirm that the levels within the testing data is the same as the training
}

#Impute levels for new levels
test_data2[test_data2['MSSubClass']=="150",'MSSubClass']<- 20 #Assigns 20 levels to the testing data for with the levels are not matching

#Predict test data
pred_test1 <- predict(lin_model2, newdata = test_data2) #predict based on the linear model in the testing data

#Transform prediction to exponential
pred_test1 <- sapply(pred_test1, exp) #applies exponential transformation to the prediction testing

#cbind Id and prediction
submission_df <- cbind(test_data['Id'], pred_test1) #combine the columns from the prediction testing by ID
colnames(submission_df) <- c("Id", "SalePrice") #subset ID and SalePrice to the submission object
submission_df$SalePrice[submission_df$SalePrice<0]=median(submission_df$SalePrice) #For all submissions that are zerio, compute the median

#Export to csv
write.csv(submission_df, file = "Competition_1-4_house_price_lm.csv", row.names = F)

# ---------------------- Building a Random Forrest Model to Answer the Kaggle Problem

#Build random forest model using same variables as earlier. The ensemble will have 50 trees
rf_model1 <- randomForest(SalePrice ~ ., data = train_data3, ntree = 50)

#Get list of factor variables
factor_vars_test_rf <- sapply(test_data2, is.factor) #list the variables that are factors in the testing data 2

#Impute levels for factor variables
#utilize the impute function we creates to define missing values for testing data
test_data2 <- impute_level(train_data3, test_data2, colnames(test_data2[factor_vars_test_rf])) 

#Ensure test data has same levels as train data
for(i in 1:ncol(test_data2)){ #for column in all the columns in the testing data
  levels(test_data2[,i]) <- levels(train_data3[,i]) #convert columns to assignment of training to testing
}

#Convert data type from numeric to integer for random forest
test_data2$BsmtFinSF1 <- as.integer(test_data2$BsmtFinSF1) #BsmtFinSF1
test_data2$LotFrontage <- as.integer(test_data2$LotFrontage) #LotFrontage
test_data2$BsmtUnfSF <- as.integer(test_data2$BsmtUnfSF) #BsmtUnfSF

pred_rf <- predict(rf_model1, test_data2) #Predict test data using random forest model

pred_rf <- exp(pred_rf) #Convert predicted price to exponential

#Combine Id and Predicted price
submission_rf_df <- cbind(test_data['Id'], pred_rf) #combined test data ID and predictions to a new object
colnames(submission_rf_df) <- c("Id", "SalePrice") #names the columns as ID and sale price per submission requirements

# Export to csv
write.csv(submission_rf_df, file = "Competition_1-4_house_price_rf.csv", row.names = F)

# ---------------------- Combine Linear Model and Random Forrest Results to Answer the Kaggle Problem

#Build ensemble model using both linear model and random forest
lm_rf_ensemble = submission_df

# Take the average predicted price from linear model and random forest
lm_rf_ensemble$SalePrice <- (submission_df$SalePrice +  submission_rf_df$SalePrice)/2

# Export to csv 
write.csv(lm_rf_ensemble, file = "Competition_1-4_house_price_rf_lm_ensemble.csv", row.names = F)

# ---------------------- Non-Parametric solution with KNN Regression Model to Answer the Kaggle Problem

#Preferably used - read in original training and testing datasets
train.url <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-house-prices/master/train.csv')
test.url <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-house-prices/master/test.csv')
master.train <- read.csv(text = train.url)
master.test <- read.csv(text = test.url) 

#Option to read in training and testing dataset that were created and imputed from above
#master.test <- test_data2 
#master.train <- train_data2

#split the train data 50-50 into train and valid
library("caret")
set.seed(500) #set the number of samples at 500
inTrain<-createDataPartition(y=master.train$SalePrice, p=0.70, list=FALSE)
train.data<-master.train[inTrain,] #subet the data by row
test.data<-master.train[-inTrain,] #subet the data by all that was not included in the first subset
nrow(train.data) #number of rows are 1024
nrow(test.data) #number of rows

##Build and evaluate KNN model using all the variables-- Model 1
model_knn <- IBk(SalePrice~., data=train.data, control=Weka_control(K=10, X=TRUE)) #create the model with K=10
model_knn #view the knn model
summary(model_knn) #provides a summary of the testing model
#evaluation function of the model
evaluate_Weka_classifier(model_knn,test.data, numFolds=10, complexity= FALSE, seed=1, class=TRUE)

#Cross-validation for IBk
model_knn <- IBk(SalePrice~., data=master.train, control=Weka_control(K=15, X=TRUE)) #set k=15 using the master.train file
evaluate_Weka_classifier(model_knn,master.train, numFolds=2, complexity= FALSE, seed=1, class=TRUE)

#prediction
prediction_knn <-predict(model_knn, master.test) #creates prediction of sale price using the KNN model
summary(prediction_knn)

#write to csv file
kaggle.submission = cbind(test_data['Id'], prediction_knn) #writes predictions with an ID column to a new object
colnames(kaggle.submission) = c("Id", "SalePrice") #establishes column names for kaggle submission
write.csv(kaggle.submission, file = "Team1-4_kaggle_house_submission3.csv", row.names = FALSE) #writes to a CSV

##Build and evaluate KNN model using most corelated variables-- Model 2
master.train2 <- master.train[,c('MSZoning','LotFrontage','LotArea','Street','Utilities','Neighborhood','OverallCond','YearBuilt','CentralAir','Fireplaces','GarageArea','GarageCars','PoolArea','YrSold','SaleType','SaleCondition','OverallQual','SalePrice')]
master.pred2 <- master.test[,c('MSZoning','LotFrontage','LotArea','Street','Utilities','Neighborhood','OverallCond','YearBuilt','CentralAir','Fireplaces','GarageArea','GarageCars','PoolArea','YrSold','SaleType','SaleCondition','OverallQual')]

#split the train data 50-50 into train and valid
library("caret") 
set.seed(500) #set the number of samples at 500
inTrain2<-createDataPartition(y=master.train2$SalePrice, p=0.50, list=FALSE)
train.data2<-master.train2[inTrain2,] #subet the data by selected rows
test.data2<-master.train2[-inTrain2,] #subet the data by all that was not included in the first subset
nrow(train.data2) #number of rows calculated
nrow(test.data2)

##Build and evaluate KNN model using all the variables-- Model 1
model_knn2 <- IBk(SalePrice~., data=train.data2, control=Weka_control(K=20, X=TRUE)) #set K=20
model_knn2 #producing the second KNN model
summary(model_knn2) #providing a summary of the model
evaluate_Weka_classifier(model_knn,test.data2, numFolds=0, complexity= FALSE, seed=1, class=TRUE) #evaluation function

#Cross-validation for IBk
model_knn2 <- IBk(SalePrice~., data=master.train2) #cross validating variables in model to training set 2
evaluate_Weka_classifier(model_knn,master.train2, numFolds=0, complexity= FALSE, seed=1, class=TRUE)

#prediction
prediction_knn2 <-predict(model_knn2, master.pred2) #gives prediction of sale price for KNN Model
summary(prediction_knn2) #provides a summary of second model

#write to csv file
kaggle.submission2 = cbind(master.test[,'Id'], prediction_knn2)
colnames(kaggle.submission2) = c("Id", "SalePrice")
write.csv(kaggle.submission2, file = "Team1-4_kaggle_house_submission2.csv", row.names = FALSE)