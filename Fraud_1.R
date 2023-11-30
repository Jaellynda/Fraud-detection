## Jael Tegulwa
# Fraud Detection
## Goal : Use machine learning to accurately identify fradulent transactions

## The Data 
# The dataset used is from https://www.kaggle.com/datasets/ealaxi/paysim1/data
# Paysim supplied artificial data for mobile money transactions spanning one month. This dataset comprises 11 attributes, listed below, and approximately 6.3 million entries. An overview of the variable properties is presented here:

## Data Preprocessing

# Load Libraries and import data
library(dplyr)
library(tidyverse)
library(ggplot2)
transactions<-read.csv("fraud.csv", header=TRUE, sep=",")
 
# View the data
head(transactions)

dim(transactions)
# The datset contains 11 attributes and 6,362,620 transaction observations

# Let's view a summary of the data. 
summary(transactions)

#These summary statistics provide a quick overview of the data in each column, including measures of central tendency (like mean and median), measures of spread (like range and quartiles), and other relevant information.
#type" is a character variable, so it shows the length, class, and mode of the values in this column.
#"amount" is another numerical variable with similar summary statistics.
#"nameOrig" is a character variable, so it also displays length, class, and mode.
#"oldbalanceOrg" and "newbalanceOrig" are numerical variables with their summary statistics.
#"nameDest" is a character variable, showing length, class, and mode.
#"oldbalanceDest" is a numerical variable with its summary statistics.
#"newbalanceDest" is another numerical variable with summary statistics.
#"isFraud" is a binary variable (0 or 1) with its minimum, 1st quartile, median, mean, 3rd quartile, and maximum values.
#"isFlaggedFraud" is a binary variable (0 or 1) with its minimum, 1st quartile, median, mean, 3rd quartile, and maximum values.

## Exploratory Data Analysis

## How many transactions are fraudulent?

# Using the isFraud variable, number of fraud vs not fraud transactions
fraud_count<- transactions %>% count(isFraud)
print(fraud_count)

## Visualizations of fraud frequency

# Lets visualize this through a frequency plot
g <- ggplot(transactions, aes(x = factor(isFraud)))
g + geom_bar() +
  geom_text(stat = 'count', aes(label = scales::percent((..count..) / sum(..count..), accuracy = 0.1)))

#The dataset exhibits a significant imbalance, with fraudulent transactions accounting for a mere 0.1% of the total data. To mitigate potential model bias, it's advisable to explore sampling techniques, such as undersampling, as a preliminary step before moving on to the modeling phase

## Which categories of transactions are associated with fraudulent activities?

ggplot(data = transactions, aes(x = type , fill = as.factor(isFraud))) + geom_bar() + labs(title = "Frequency of Transaction Type", subtitle = "Fraud vs Not Fraud", x = 'Transaction Type' , y = 'No of transactions' ) +theme_classic()

# The plot displayed above may not offer much insight into fraud transactions due to their infrequency. Let's focus on creating a plot specifically for transaction types in cases of fraud.

## Transaction Types for Fraudulent cases

## ggplot showing frequency of Fraud Transactions for each Transaction type

Fraud_transaction_type <- transactions %>% group_by(type) %>% summarise(fraud_transactions = sum(isFraud))
ggplot(data = Fraud_transaction_type, aes(x = type,  y = fraud_transactions)) + geom_col(aes(fill = 'type'), show.legend = FALSE) + labs(title = 'Fraud transactions as Per type', x = 'Transcation type', y = 'No of Fraud Transactions') + geom_label(aes(label = fraud_transactions)) + theme_classic()


# As observed in the plot above, fraud transactions are exclusively associated with the "CASH_OUT" and "TRANSFER" transaction types. This observation will be significant in our subsequent analysis, as we can streamline our assessment by concentrating solely on these two transaction types.

## How is the distribution of transaction amounts distributed in cases of fraud?

ggplot(data = transactions[transactions$isFraud==1,], aes(x = amount ,  fill =amount)) + geom_histogram(bins = 30, aes(fill = 'amount')) + labs(title = 'Fraud transaction Amount distribution', y = 'No. of Fraud transacts', x = 'Amount in Dollars')

# The distribution of transaction amounts for fraudulent cases exhibits a pronounced right skew. This indicates that the majority of fraudulent transactions involve smaller amounts.

## oldbalanceOrg vs oldbalanceDest

library(gridExtra)

p1 <- ggplot(data = transactions, aes(x = factor(isFraud), y = log1p(oldbalanceOrg), fill = factor(isFraud))) + 
  geom_boxplot(show.legend = FALSE) +
  labs(title = 'Old Balance in Sender Accounts', x = 'isFraud', y = 'Balance Amount') +
  theme_classic()

p2 <- ggplot(data = transactions, aes(x = factor(isFraud), y = log1p(oldbalanceDest), fill = factor(isFraud))) + 
  geom_boxplot(show.legend = FALSE) +
  labs(title = 'Old Balance in Receiver Accounts', x = 'isFraud', y = 'Balance Amount') +
  theme_classic()

grid.arrange(p1, p2, nrow = 1)

# For most fraudulent transactions, the initial account's old balance (where payments originate) tends to be higher than the old balances of other origin accounts, while the old balance in destination accounts is lower than the rest.

## Is there a specific time of day when fraud incidents are more prevalent?
# Convert Step to Hours in 24 hours format
transactions$hour <- transactions$step %% 24

# Plot newly formatted data
p5<- ggplot(data = transactions, aes(x = hour)) + geom_bar(aes(fill = 'isFraud'), show.legend = FALSE) +labs(title= 'Total transactions at different Hours', y = 'No. of transactions') + theme_classic()

p6<-ggplot(data = transactions[transactions$isFraud==1,], aes(x = hour)) + geom_bar(aes(fill = 'isFraud'), show.legend = FALSE) +labs(title= 'Fraud transactions at different Hours', y = 'No. of fraud transactions') + theme_classic()

grid.arrange(p5, p6, ncol = 1, nrow = 2)

# The overall transaction count during the 0 to 9-hour period is notably low. However, this pattern doesn't hold true for fraudulent transactions. It can be inferred that fraudulent activities are more frequent from midnight to 9 am.

## Key Insights to Keep in Mind:

# The dataset exhibits a significant class imbalance in terms of target classes. To mitigate model bias, it's advisable to explore sampling techniques, such as undersampling. 
# Focusing on transaction types like CASH_OUT and TRANSFER is recommended, as these are the only transaction types associated with fraudulent cases.
# Fraudulent transactions typically involve smaller monetary amounts.
# Fraudulent activities are more prevalent during the early hours, from midnight to 9 am.

## Feature Engineering and Data Cleaning

#Verify if transactions occur where the transaction amount exceeds the available balance in the origin account.

head(transactions[(transactions$amount > transactions$oldbalanceOrg)& (transactions$newbalanceDest > transactions$oldbalanceDest), c("amount","oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud")], 10)

## Data Filtering

#I will filter the data by transaction type to include only CASH_OUT and TRANSFER. Additionally, we can exclude the columns "nameOrig" and "nameDest" due to the excessive number of unique levels, which would make it impractical to create dummy variables for them. Moreover, we can safely remove the "step" column since it was utilized to derive the "hour" attribute.

# Filtering transactions and drop irrelevant features
transactions1<- transactions %>% 
  select( -one_of('step','nameOrig', 'nameDest', 'isFlaggedFraud')) %>%
  filter(type %in% c('CASH_OUT','TRANSFER'))


# Every fraudulent transaction falls within the CASH_OUT and TRANSFER types, making other types irrelevant. Since the "step" column was used to generate the "hour" variable, we can safely discard it. Additionally, "nameOrig," "nameDest," and "isFlaggedFraud" serve no significant purpose and can be removed.

## Encoding Dummy variables for transaction type

library(fastDummies)

transactions1 <- dummy_cols(transactions1)

transactions1$isFraud <- as.factor(transactions1$isFraud)
transactions1 <- transactions1[,-1]
transactions1<-as.data.frame(transactions1)
#summarizeColumns(transactions1)

## Train and Test Data

# We understand that fraudulent transactions are infrequent. Given that fraudulent transactions account for only 0.1% of the dataset, employing duplication to balance the data is not the most effective approach. Instead, a more sensible strategy is to reduce the number of non-fraud cases through undersampling.

set.seed(12345)
train_id <- sample(seq_len(nrow(transactions1)), size = floor(0.7*nrow(transactions1)))

train <- transactions1[train_id,]
valid <- transactions1[-train_id,]

table(train$isFraud)

table(valid$isFraud)

## Undersampling

suppressMessages(library(ROSE))
set.seed(12345)
prop.table(table(train$isFraud))

inputs <- train[,-6]
target <- train[,6]
# Downsample the data
down_train <- downSample(x = inputs, y = target)

# Calculate the proportions
proportions <- prop.table(table(down_train$Class))

# Print the proportions
print(proportions)


# To see it as percentages:
percentages <- prop.table(table(down_train$Class)) * 100
print(percentages)

# By applying undersampling, we achieve a balanced training dataset consisting of 5,707 observations for each class. Nevertheless, it's important to note that our validation dataset will still maintain its imbalanced nature.

## Modeling
# We explore tree-based modeling approaches like Decision Trees and Random Forests for predicting fraudulent transactions.


## Decision tree
# Using the undersampled data

# Load the rpart package for decision trees
# Load necessary libraries
library(rpart)
library(rpart.plot) # for visualizing the decision tree

# Build the decision tree model using the balanced data 'down_train'
# and using 'Class' as the dependent variable
model_dt <- rpart(Class ~ ., data = down_train)

# Plot the decision tree
prp(model_dt)  # prp function is from the rpart.plot package

predict_dt <- predict(model_dt, valid, type = "class")
head(predict_dt)

confusionMatrix(valid$isFraud,predict_dt)


# Accuracy of the model is 93.6%

## Calculate the F1 score

# The F1 score serves as a weighted average of Precision and Recall. We can compute this metric using the provided information because:

# Recall is synonymous with Sensitivity,
# Precision corresponds to Positive Predictive Value.

# Recall=Sensitivity
Recall_1<-0.99985
Precision_1<-0.93595

F1<-2*(Recall_1*Precision_1)/(Recall_1+Precision_1)
F1*100


# The F1 score for the decision tree model is 96.68% which is very good.

## Random Forest

# Create a Random Forest model with default parameters
library(randomForest)

# Create a Random Forest model with default parameters using the down_train data frame
rf1 <- randomForest(Class ~ ., data = down_train, importance = TRUE)

# Print the model output to see the results
print(rf1)



#The estimated error rate is currently at 2.04%. We will explore if we can enhance this by adjusting the model parameters. In random forest models, the key parameters include:

#ntree: The number of trees in the forest.
#mtry: The number of predictor variables included in each split.

rf2 <- randomForest(Class ~ ., data = down_train, ntree = 500, mtry = 6, importance = TRUE)
rf2


#The estimated error rate has improved to 1%, which is better than the first model. Now, let's explore one more random forest scenario. In this model, we will reduce the number of trees in the forest but maintain the same number of predictors included for the node split

rf3 <- randomForest(Class ~ ., data = down_train, ntree = 200, mtry = 6, importance = TRUE)
rf3


#The estimated error rate in this scenario is 1.03%, which is slightly higher than in model 2. We will proceed to use model 2 to evaluate our validation dataset. In this step, we assess the predictive performance of our random forest model on the training dataset

# Predicting on train set
predTrain <- predict(rf2, train, type = "class")
# Checking classification accuracy
table(predTrain, train$isFraud)


# The model demonstrates good performance on the training dataset, and now we'll assess its effectiveness on the validation dataset.

# Predicting on Validation set
predValid <- predict(rf2, valid, type = "class")
# Checking classification accuracy
mean(predValid == valid$isFraud)  


table(predValid,valid$isFraud)


#The random forest model exhibited strong performance with an accuracy of 98.7%. Now, let's compute the F1 score for this model.

TP=817876 
TN=2487
FP=10741
FN=19
#Calculate Recall
Recall=(TP/(TP+FN))
Recall*100

# Calculate Precision
Precision=(TP/(TP+FP))
# Calculate F1
F1_rf<-2*(Recall_1*Precision_1)/(Recall_1+Precision_1)
F1_rf*100


# The random forest model has an f1 score of 96.68%

# To check important variables
importance(rf2) 


varImpPlot(rf2)

# The plots above provide insights into the variable importance in our random forest model, ranking them from most to least significant when making predictions. This aids in addressing the interpretability concerns associated with random forest models.

## Model Conclusions

# Our decision tree model has:
#  accuracy of 93.6%
# f1 score of 96.68%
# Recall of 99.998

# Our random forest model has:
#  accuracy of 98.7%
# f1 score of 96.68%
# Recall of 99.999

# Typically, in this situation, the F1 score is used to determine the preferred model since it offers a balanced assessment of Precision and Recall. However, both models have the same F1 score. While the Recall is only 0.001 higher for the random forest model compared to the decision tree model, we will consider this as our scoring metric. The reason for choosing Recall over accuracy, even though both are viable, is due to the higher cost associated with False Negatives. Given that the company faces greater consequences from failing to detect a fraudulent transaction, it makes Recall the more suitable choice. Consequently, the company should opt for the random forest model for predicting fraud.

## Conclusion

# Paysim provided simulated data for fraudulent mobile money transactions. The objective of this project was to achieve accurate predictions of fraudulent transactions using tree-based machine learning algorithms. Surprisingly, both models surpassed expectations, possibly attributed to the characteristics of the simulated data.
## Suggested Next Steps

#Explore alternative methods for fine-tuning model parameters. Additionally, consider employing clustering techniques to enhance the performance of the random forest model.
#Investigate the utilization of other machine learning techniques such as XGBoost and Support Vector Machine algorithms, which may offer valuable insights and improve predictive capabilities.
#Delve deeper into the reasons behind fraudulent transactions primarily occurring within TRANSFER and CASH_OUT transaction types, providing a more comprehensive understanding of the data's underlying patterns and characteristics.

