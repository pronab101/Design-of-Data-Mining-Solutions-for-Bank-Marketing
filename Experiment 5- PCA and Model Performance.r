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
library(tidyr) 

# ----------------------------------------------
# Load the Dataset
# ----------------------------------------------
Mydata <- read.csv("/Users/pronabkarmaker/R/bank-additional.csv", sep = ";")

# Preprocess the data 
Mydata$y <- as.factor(Mydata$y)  # Assuming 'y' is the target variable

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(Mydata$y, p = .8, list = FALSE)
trainData <- Mydata[trainIndex, ]
testData <- Mydata[-trainIndex, ]


# ----------------------------------------------
# Experiment 4: PCA and Model Performance
# ----------------------------------------------

# Apply PCA on the dataset
trainData_pca <- trainData[, -which(names(trainData) == "y")]
testData_pca <- testData[, -which(names(testData) == "y")]

# Ensure numeric data
trainData_pca <- trainData_pca[sapply(trainData_pca, is.numeric)]
testData_pca <- testData_pca[sapply(testData_pca, is.numeric)]

# Standardize
trainData_pca_scaled <- scale(trainData_pca)
testData_pca_scaled <- scale(testData_pca, center = attr(trainData_pca_scaled, "scaled:center"), scale = attr(trainData_pca_scaled, "scaled:scale"))

# PCA
pca_model <- prcomp(trainData_pca_scaled, center = TRUE, scale. = TRUE)
pca_summary <- summary(pca_model)

# Variance plot
pca_var_plot <- ggplot(data.frame(Variance = pca_summary$importance[2, ]), aes(x = 1:length(Variance), y = Variance)) +
  geom_line() +
  geom_point() +
  labs(title = "Variance Explained by Principal Components", x = "Principal Component", y = "Proportion of Variance Explained") +
  theme_minimal()

# Components explaining 95% variance
cumulative_variance <- cumsum(pca_summary$importance[2, ])
num_components <- which(cumulative_variance >= 0.95)[1]

# Transform data
trainData_pca_transformed <- data.frame(pca_model$x[, 1:num_components])
testData_pca_transformed <- data.frame(predict(pca_model, newdata = testData_pca_scaled)[, 1:num_components])

# Add target variable back
trainData_pca_transformed$y <- trainData$y
testData_pca_transformed$y <- testData$y

# Random Forest with PCA
rf_model_pca <- randomForest(y ~ ., data = trainData_pca_transformed, ntree = 100)
rf_predictions_pca <- predict(rf_model_pca, testData_pca_transformed)
rf_confusion_matrix_pca <- confusionMatrix(rf_predictions_pca, testData_pca_transformed$y)

# ANN with PCA
ann_model_pca <- nnet(y ~ ., data = trainData_pca_transformed, size = 5, maxit = 100, linout = FALSE, trace = FALSE)
ann_predictions_pca <- predict(ann_model_pca, testData_pca_transformed, type = "class")
ann_predictions_pca <- factor(ann_predictions_pca, levels = levels(testData_pca_transformed$y))
ann_confusion_matrix_pca <- confusionMatrix(ann_predictions_pca, testData_pca_transformed$y)

# Compile results in a data frame with TP, TN, FP, FN
pca_results <- data.frame(
  Model = c("Random Forest with PCA", "ANN with PCA"),
  Accuracy = c(rf_confusion_matrix_pca$overall['Accuracy'],
               ann_confusion_matrix_pca$overall['Accuracy']),
  ErrorRate = 1 - c(rf_confusion_matrix_pca$overall['Accuracy'],
                    ann_confusion_matrix_pca$overall['Accuracy']),
  TP = c(rf_confusion_matrix_pca$table[2, 2], ann_confusion_matrix_pca$table[2, 2]),
  TN = c(rf_confusion_matrix_pca$table[1, 1], ann_confusion_matrix_pca$table[1, 1]),
  FP = c(rf_confusion_matrix_pca$table[1, 2], ann_confusion_matrix_pca$table[1, 2]),
  FN = c(rf_confusion_matrix_pca$table[2, 1], ann_confusion_matrix_pca$table[2, 1])
)

print(pca_results)

# Plot variance and model comparison including TP, TN, FP, FN counts
gridExtra::grid.arrange(
  pca_var_plot,
  ggplot(pca_results, aes(x = Model, y = ErrorRate, fill = Model)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Error Rates with PCA", y = "Error Rate", x = "Model"),
  ggplot(pca_results %>% pivot_longer(cols = c("TP", "TN", "FP", "FN"), names_to = "Metric", values_to = "Count"), 
         aes(x = Model, y = Count, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "Confusion Matrix Elements for PCA Models", y = "Count", x = "Model") +
    scale_fill_brewer(palette = "Set1", name = "Element"),
  nrow = 1
)
