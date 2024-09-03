# ----------------------------------------------
# Libraries
# ----------------------------------------------
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(tidyr)
library(gridExtra)
library(lattice)
# ----------------------------------------------
# Load the Dataset
# ----------------------------------------------
Mydata <- read.csv("/Users/pronabkarmaker/R/bank-additional.csv", sep = ";")

# Preprocess the data 
Mydata$y <- as.factor(Mydata$y)  # Assuming 'y' is the target variable

# Parameters
split_size <- 0.7   # Split size (80% training, 20% testing)
cp_value <- 0.1     # Complexity parameter
max_depth <- 5     # Maximum depth of the tree

# Custom Error Cost Matrix (higher cost for False Negatives)
error_costs <- matrix(c(0, 1, 10, 0), nrow = 2, dimnames = list(c("no", "yes"), c("no", "yes")))

# Data Splitting
set.seed(42)
trainIndex <- createDataPartition(Mydata$y, p = split_size, list = FALSE)
trainData <- Mydata[trainIndex, ]
testData <- Mydata[-trainIndex, ]

# Decision Tree Model
dt_control <- rpart.control(cp = cp_value, maxdepth = max_depth)
dt_model <- rpart(y ~ ., data = trainData, method = "class", parms = list(loss = error_costs), control = dt_control)
dt_predictions <- predict(dt_model, testData, type = "class")
dt_confusion_matrix <- confusionMatrix(dt_predictions, testData$y)

# Extracting and printing results
TP <- dt_confusion_matrix$table[2, 2]
TN <- dt_confusion_matrix$table[1, 1]
FP <- dt_confusion_matrix$table[1, 2]
FN <- dt_confusion_matrix$table[2, 1]

# Compile results in a data frame
dt_results <- data.frame(
  Metric = c("True Positives", "True Negatives", "False Positives", "False Negatives"),
  Count = c(TP, TN, FP, FN)
)

# Accuracy and Error Rate
accuracy <- dt_confusion_matrix$overall['Accuracy']
error_rate <- 1 - accuracy


# Print Parameters and Results
cat("Example 1: Basic Configuration\n")
cat("Parameters:\n")
cat("  Split Size: ", split_size, "\n")
cat("  Complexity Parameter (cp): ", cp_value, "\n")
cat("  Maximum Depth: ", max_depth, "\n")
cat("  Error Cost Matrix:\n")
print(error_costs)
cat("\nModel Performance:\n")
cat("  Accuracy: ", round(accuracy, 4), "\n")
cat("  Error Rate: ", round(error_rate, 4), "\n")
cat("\nConfusion Matrix Elements:\n")
print(data.frame(Metric = c("True Positives", "True Negatives", "False Positives", "False Negatives"), 
                  Count = c(TP, TN, FP, FN)))

# ----------------------------------------------
# Plotting
# ----------------------------------------------

# Confusion Matrix Elements Plot
p1 <- ggplot(dt_results, aes(x = Metric, y = Count, fill = Metric)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Decision Tree Confusion Matrix Elements", y = "Count", x = "Metric") +
  scale_fill_brewer(palette = "Set1")

# Accuracy and Error Rate Plot
p2 <- ggplot(data.frame(Metric = c("Accuracy", "Error Rate"), Value = c(accuracy, error_rate)), 
             aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Decision Tree Accuracy and Error Rate", y = "Value", x = "Metric") +
  scale_fill_brewer(palette = "Set1")

# Display both plots
grid.arrange(p1, p2, ncol = 1)
