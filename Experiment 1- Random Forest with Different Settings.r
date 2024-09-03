# ----------------------------------------------
# Libraries
# ----------------------------------------------
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(nnet)
library(gridExtra)
library(tidyr)  # For pivot_longer

# ----------------------------------------------
# Load the Dataset
# ----------------------------------------------
Mydata <- read.csv("/Users/pronabkarmaker/R/bank-additional.csv", sep = ";")

# Preprocess the data (convert factors if needed)
Mydata$y <- as.factor(Mydata$y)  # Assuming 'y' is the target variable

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(Mydata$y, p = .8, list = FALSE)
trainData <- Mydata[trainIndex, ]
testData <- Mydata[-trainIndex, ]

# ----------------------------------------------
# Experiment 1: Random Forest with Different Settings
# ----------------------------------------------

# Setting 1: Default (100 trees, default mtry)
set.seed(42)
rf_model_100_trees <- randomForest(y ~ ., data = trainData, ntree = 100)
varImpPlot(rf_model_100_trees)
rf_predictions_default <- predict(rf_model_100_trees, testData)
rf_confusion_matrix_default <- confusionMatrix(rf_predictions_default, testData$y)

# Extracting confusion matrix elements
TP_default <- rf_confusion_matrix_default$table[2, 2]
TN_default <- rf_confusion_matrix_default$table[1, 1]
FP_default <- rf_confusion_matrix_default$table[1, 2]
FN_default <- rf_confusion_matrix_default$table[2, 1]

# Setting 2: 50 Trees
set.seed(42)
rf_model_50_trees <- randomForest(y ~ ., data = trainData, ntree = 50)
varImpPlot(rf_model_50_trees)
rf_predictions_50 <- predict(rf_model_50_trees, testData)
rf_confusion_matrix_50 <- confusionMatrix(rf_predictions_50, testData$y)

# Extracting confusion matrix elements
TP_50 <- rf_confusion_matrix_50$table[2, 2]
TN_50 <- rf_confusion_matrix_50$table[1, 1]
FP_50 <- rf_confusion_matrix_50$table[1, 2]
FN_50 <- rf_confusion_matrix_50$table[2, 1]

# Setting 3: mtry = 2
set.seed(42)
rf_model_mtry <- randomForest(y ~ ., data = trainData, ntree = 100, mtry = 2)
varImpPlot(rf_model_mtry)
rf_predictions_2_vars <- predict(rf_model_mtry, testData)
rf_confusion_matrix_2_vars <- confusionMatrix(rf_predictions_2_vars, testData$y)

# Extracting confusion matrix elements
TP_2_vars <- rf_confusion_matrix_2_vars$table[2, 2]
TN_2_vars <- rf_confusion_matrix_2_vars$table[1, 1]
FP_2_vars <- rf_confusion_matrix_2_vars$table[1, 2]
FN_2_vars <- rf_confusion_matrix_2_vars$table[2, 1]

# Setting 4: Subset of 1000 samples, with adjusted mtry and sampsize
set.seed(42)
rf_model_sampled <- randomForest(y ~ ., data = trainData, ntree = 1000, mtry = 10, sampsize = c(500, 500))
varImpPlot(rf_model_sampled)
rf_predictions_sampled <- predict(rf_model_sampled, testData)
rf_confusion_matrix_sampled <- confusionMatrix(rf_predictions_sampled, testData$y)

# Extracting confusion matrix elements
TP_sampled <- rf_confusion_matrix_sampled$table[2, 2]
TN_sampled <- rf_confusion_matrix_sampled$table[1, 1]
FP_sampled <- rf_confusion_matrix_sampled$table[1, 2]
FN_sampled <- rf_confusion_matrix_sampled$table[2, 1]

# Compile results in a data frame
rf_results <- data.frame(
  Setting = c("100 trees", "50 Trees", "mtry = 2", "Sampled Data"),
  Accuracy = c(rf_confusion_matrix_default$overall['Accuracy'],
               rf_confusion_matrix_50$overall['Accuracy'],
               rf_confusion_matrix_2_vars$overall['Accuracy'],
               rf_confusion_matrix_sampled$overall['Accuracy']),
  ErrorRate = 1 - c(rf_confusion_matrix_default$overall['Accuracy'],
                    rf_confusion_matrix_50$overall['Accuracy'],
                    rf_confusion_matrix_2_vars$overall['Accuracy'],
                    rf_confusion_matrix_sampled$overall['Accuracy']),
  TP = c(TP_default, TP_50, TP_2_vars, TP_sampled),
  TN = c(TN_default, TN_50, TN_2_vars, TN_sampled),
  FP = c(FP_default, FP_50, FP_2_vars, FP_sampled),
  FN = c(FN_default, FN_50, FN_2_vars, FN_sampled)
)

print(rf_results)

# Plot error rates
p1 <- ggplot(rf_results, aes(x = Setting, y = ErrorRate, fill = Setting)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Random Forest Error Rates for Different Settings", y = "Error Rate", x = "Setting")

# Plot TP, TN, FP, FN counts using pivot_longer
rf_results_long <- rf_results %>%
  pivot_longer(cols = c("TP", "TN", "FP", "FN"), names_to = "Metric", values_to = "Count")

p2 <- ggplot(rf_results_long, aes(x = Setting, y = Count, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Confusion Matrix Elements for Different Settings", y = "Count", x = "Setting") +
  scale_fill_brewer(palette = "Set1", name = "Element")

# Arrange both plots together
grid.arrange(p1, p2, ncol = 1)
