# Load required libraries
library(neuralnet)
library(caret)  # For confusion matrix

# Load your dataset
bank_data <- read.csv("/Users/pronabkarmaker/R/bank-additional.csv", sep = ";")

# Apply normalization to relevant numeric columns
numeric_cols <- c("age", "campaign", "pdays", "cons.conf.idx", "emp.var.rate", "euribor3m", "nr.employed")
bank_data[numeric_cols] <- scale(bank_data[numeric_cols])

# Convert the target variable 'y' to a factor with levels "yes" and "no"
bank_data$y <- factor(bank_data$y, levels = c("no", "yes"))

# Split data into training (80%) and validation (20%)
set.seed(123)
A <- sort(sample(nrow(bank_data), nrow(bank_data) * 0.8))
Train <- bank_data[A, ]
Val <- bank_data[-A, ]

# Train a neural network using the neuralnet package
ann_model <- neuralnet(y ~ age + campaign + pdays + cons.conf.idx + emp.var.rate + euribor3m + nr.employed, 
                       data = Train, hidden = 5, linear.output = FALSE, stepmax = 1e5)

# Check if the neural network has completed the training process
if (length(ann_model$weights) == 0) {
  stop("The neural network did not converge. Please try increasing the stepmax or adjusting the model complexity.")
}

# Visualize the neural network
plot(ann_model)

# Predict on validation data using the model
predict_columns <- c("age", "campaign", "pdays", "cons.conf.idx", "emp.var.rate", "euribor3m", "nr.employed")
Val_subset <- Val[, predict_columns]

# Compute predictions
ann_predictions_raw <- compute(ann_model, Val_subset)$net.result

# Check the dimensions of the prediction output
cat("Dimensions of raw predictions: ", dim(ann_predictions_raw), "\n")

# Ensure the output is a single-column vector
if (is.matrix(ann_predictions_raw) && ncol(ann_predictions_raw) == 1) {
  ann_predictions <- ifelse(ann_predictions_raw > 0.5, "yes", "no")
} else if (is.matrix(ann_predictions_raw) && ncol(ann_predictions_raw) > 1) {
  # If more than one column is returned, assume the second column corresponds to the "yes" class
  ann_predictions <- ifelse(ann_predictions_raw[, 2] > 0.5, "yes", "no")
} else {
  stop("Unexpected dimensions in prediction output. Expected a single column vector.")
}

# Convert predictions to a factor with levels "yes" and "no"
ann_predictions <- factor(ann_predictions, levels = c("no", "yes"))

# Check the lengths of predictions and validation labels
cat("Length of Predictions: ", length(ann_predictions), "\n")
cat("Length of Validation Labels: ", length(Val$y), "\n")

if (length(ann_predictions) != length(Val$y)) {
  stop("Error: Lengths of predictions and validation labels do not match.")
}

# Generate the confusion matrix
ann_confusion_matrix <- confusionMatrix(ann_predictions, Val$y)

# Extract metrics
accuracy <- ann_confusion_matrix$overall['Accuracy']
error_rate <- 1 - accuracy
TP <- ann_confusion_matrix$table["yes", "yes"]
TN <- ann_confusion_matrix$table["no", "no"]
FP <- ann_confusion_matrix$table["no", "yes"]
FN <- ann_confusion_matrix$table["yes", "no"]

# Print results
cat("Neural Network Model Evaluation:\n")
cat("Accuracy: ", accuracy, "\n")
cat("Error Rate: ", error_rate, "\n")
cat("True Positives (TP): ", TP, "\n")
cat("True Negatives (TN): ", TN, "\n")
cat("False Positives (FP): ", FP, "\n")
cat("False Negatives (FN): ", FN, "\n")
