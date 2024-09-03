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
# Experiment 2: Decision Tree vs. Random Forest with Different Parameters
# ----------------------------------------------

# Decision Tree Model with Different Parameters
dt_control <- rpart.control(minsplit = 20,     # Minimum number of observations in a node for a split
                            cp = 0.01,         # Complexity parameter for pruning
                            maxdepth = 30)         # Number of cross-validation runs

dt_model <- rpart(y ~ ., data = trainData, method = "class", control = dt_control)
dt_predictions <- predict(dt_model, testData, type = "class")
dt_confusion_matrix <- confusionMatrix(dt_predictions, testData$y)

# Extract Decision Tree parameters
dt_params <- list(
  MinSplit = dt_model$control$minsplit,
  CP = dt_model$control$cp,
  MaxDepth = dt_model$control$maxdepth
)

# Extracting confusion matrix elements for Decision Tree
TP_dt <- dt_confusion_matrix$table[2, 2]
TN_dt <- dt_confusion_matrix$table[1, 1]
FP_dt <- dt_confusion_matrix$table[1, 2]
FN_dt <- dt_confusion_matrix$table[2, 1]

# Random Forest Model with Different Parameters
rf_model <- randomForest(y ~ ., 
                         data = trainData, 
                         ntree = 200,             # Number of trees
                         mtry = 5            # Minimum size of terminal nodes
                    )           # Maximum number of terminal nodes

rf_predictions <- predict(rf_model, testData)
rf_confusion_matrix <- confusionMatrix(rf_predictions, testData$y)

# Extract Random Forest parameters
rf_params <- list(
  NumberOfTrees = rf_model$ntree,
  Mtry = rf_model$mtry         # Maximum number of terminal nodes
)

# Extracting confusion matrix elements for Random Forest
TP_rf <- rf_confusion_matrix$table[2, 2]
TN_rf <- rf_confusion_matrix$table[1, 1]
FP_rf <- rf_confusion_matrix$table[1, 2]
FN_rf <- rf_confusion_matrix$table[2, 1]

# Compile results and parameters into a single data frame
results_summary <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Accuracy = c(dt_confusion_matrix$overall['Accuracy'],
               rf_confusion_matrix$overall['Accuracy']),
  ErrorRate = 1 - c(dt_confusion_matrix$overall['Accuracy'],
                    rf_confusion_matrix$overall['Accuracy']),
  TP = c(TP_dt, TP_rf),
  TN = c(TN_dt, TN_rf),
  FP = c(FP_dt, FP_rf),
  FN = c(FN_dt, FN_rf)
)

# Parameters data frame with extended details
params_summary <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Parameters = c(
    paste("MinSplit:", dt_params$MinSplit, "; CP:", dt_params$CP, 
          "; MaxDepth:", dt_params$MaxDepth),
    paste("NumberOfTrees:", rf_params$NumberOfTrees, "; Mtry:", rf_params$Mtry)
  )
)

# Print parameters and results
print("Parameters Summary:")
print(params_summary)

print("Results Summary:")
print(results_summary)

# The plots can remain the same as before
p1 <- ggplot(results_summary, aes(x = Model, y = ErrorRate, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Decision Tree vs. Random Forest Error Rates", y = "Error Rate", x = "Model")

# Plot TP, TN, FP, FN counts using pivot_longer
results_summary_long <- results_summary %>%
  pivot_longer(cols = c("TP", "TN", "FP", "FN"), names_to = "Metric", values_to = "Count")

p2 <- ggplot(results_summary_long, aes(x = Model, y = Count, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Confusion Matrix Elements: Decision Tree vs. Random Forest", y = "Count", x = "Model") +
  scale_fill_brewer(palette = "Set1", name = "Element")

grid.arrange(p1, p2, ncol = 1)
