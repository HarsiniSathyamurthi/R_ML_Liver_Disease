# ----------------------------------------------------------
# SECTION 1: ENVIRONMENT SETUP
# ----------------------------------------------------------

# Clearing Environment
rm(list = ls())
dev.off()
cat("\014") 

# Set Working Directory
setwd("E:/UWA/FBA/FBA_Final_Project")


# ----------------------------------------------------------
# SECTION 2: LIBRARIES HANDLING
# ----------------------------------------------------------

# Installing Required Libraries
if(!require("tidyverse")) install.packages("tidyverse")
if(!require("caret")) install.packages("caret")
if(!require("rpart")) install.packages("rpart")
if(!require("rpart.plot")) install.packages("rpart.plot")
if(!require("randomForest")) install.packages("randomForest")
if(!require("ROCR")) install.packages("ROCR")
if(!require("naniar")) install.packages("naniar")
if(!require("ggcorrplot")) install.packages("ggcorrplot")
if(!require("ipred")) install.packages("ipred")

# Loading Required Libraries
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(naniar)
library(ggcorrplot)
library(ipred)


# ----------------------------------------------------------
# SECTION 3: DATA LOADING & CLEANING
# ----------------------------------------------------------

Liver_data <- read_csv("liver.csv")

# General Exploration of the Dataset
str(Liver_data)            # Gives the structure of the dataset
dim(Liver_data)            # Dimensions (rows, cols)
head(Liver_data)           # First few rows of the dataset
names(Liver_data)          # Displays the column names

# Assigning new column names for easy interpretation and readability
new_column_names <- c("Age","Gender","Total_Bilirubin","Direct_Bilirubin",
                      "Alkaline_Phosphatase","ALT","AST",
                      "Total_Protein","Albumin","Albumin_Globulin_Ratio",
                      "Target")
names(Liver_data) <- new_column_names
names(Liver_data)          # Displays the updated column names

# Inspecting Missing Values
colSums(is.na(Liver_data))        # Theoretical check
vis_miss(Liver_data)              # Visualize missing data patterns
Liver_data <- na.omit(Liver_data) # Handling missing values

# Visualise missing values again
vis_miss(Liver_data)

# Converting Target Variable to Factor
Liver_data$Target <- as.factor(Liver_data$Target)
Liver_data$Target <- factor(Liver_data$Target, levels = c("1", "2"), labels = c("Liver_Disease", "No_Liver_Disease"))

# Converting other Categorical Variables to Factors
factor_columns <- c("Gender")
Liver_data[factor_columns] <- lapply(Liver_data[factor_columns], as.factor)

# Ensuring Numeric Variables are correctly set as Numeric
numeric_columns <- c("Age", "Total_Bilirubin", "Direct_Bilirubin",
                     "Alkaline_Phosphatase", "ALT", "AST",
                     "Total_Protein", "Albumin", "Albumin_Globulin_Ratio")
Liver_data[numeric_columns] <- lapply(Liver_data[numeric_columns], as.numeric)

View(Liver_data)
summary(Liver_data)
# ----------------------------------------------------------
# SECTION 4: DESCRIPTIVE STATISTICS & VISUALISATIONS
# Visualisations and Graphical Analysis
# ----------------------------------------------------------

# Target Variable Distribution
table(Liver_data$Target)

# Bar plot for target variable
ggplot(Liver_data, aes(x = Target, fill = Target)) +
  geom_bar() +
  labs(title = "Class Distribution: Liver Disease vs No Liver Disease", x = "Target", y = "Count") +
  scale_x_discrete(labels = c("1" = "Liver Disease", "2" = "No Liver Disease")) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"))

# Boxplot
ggplot(Liver_data, aes(x = Target, y = ALT, fill = Target)) +
  geom_boxplot() +
  labs(title = "Boxplot: ALT Levels by Patient Condition", x = "Target", y = "ALT") +
  scale_x_discrete(labels = c("1" = "Liver Disease", "2" = "No Liver Disease")) +
  theme_minimal()


# ----------------------------------------------------------
# SECTION 5: CORRELATION ANALYSIS
# ----------------------------------------------------------

Numeric_vars <- Liver_data %>% select_if(is.numeric)    # Select only numeric columns
cor_matrix <- cor(Numeric_vars)

ggcorrplot(cor_matrix, method = "square", type = "upper", 
           lab = FALSE,
           colors = c("red", "white", "blue"),
           title = "Correlation Matrix of Numeric Features")


# ----------------------------------------------------------
# SECTION 6: DATA SPLITTING
# ----------------------------------------------------------

# Splitting the data into training and testing sets
set.seed(5555)

train_index <- createDataPartition(Liver_data$Target, p = 0.7, list = FALSE)
train <- Liver_data[train_index, ]
test <- Liver_data[-train_index, ]

# Check proportions of target variable in training and testing sets
summary(train$Target)
summary(test$Target)


# ----------------------------------------------------------
# SECTION 7: DECISION TREE MODELS
# ----------------------------------------------------------

# Building and Plotting a Decision Tree
Model_DT1 <- rpart(Target ~ ., data = train, method = "class")
rpart.plot(Model_DT1, yesno = 2, type = 0, extra = 0)

# Evaluate Model_DT1
predictions_DT1 <- predict(Model_DT1, test, type = "class")
confusion_matrix_DT1 <- confusionMatrix(predictions_DT1, test$Target, positive = "Liver_Disease")
cat("Model_DT1 Evaluation:\n")
print(confusion_matrix_DT1)

# Plot complexity parameter (CP) for pruning
plotcp(Model_DT1, main = "CP Plot for Model_DT1")

# Prune trees (Upper-level trees with cp = 0.11)
Model_DT1_pruned <- prune(Model_DT1, cp = 0.01234)  
rpart.plot(Model_DT1_pruned, yesno = 2, type = 0, extra = 0, main = "Pruned Tree (cp = 0.01234): Model_DT1")

# Generate predictions for ROC analysis
dt_probabilities <- predict(Model_DT1_pruned, test, type = "prob")[, 2]
dt_ROCR_prediction <- prediction(dt_probabilities, test$Target)
dt_ROCR_performance <- performance(dt_ROCR_prediction, "tpr", "fpr")
  
# Plot the ROC curve
plot(dt_ROCR_performance, colorize = TRUE, main = "ROC Curve for Decision Tree Pruned Model")
abline(a = 0, b = 1, col = "blue", lty = 2)

# Compute AUC
AUC_DT <- performance(dt_ROCR_prediction, "auc")@y.values[[1]]
cat("AUC for Decision Tree Model: ", AUC_DT, "\n")

# ----------------------------------------------------------
# SECTION 8: BAGGING
# ----------------------------------------------------------

set.seed(347)

# Train a bagging model
bagging_model <- bagging(Target ~ ., data = train, coob = TRUE)
print(bagging_model)

# Evaluate the bagging model
bagging_predictions <- predict(bagging_model, test, type = "class")
confusion_matrix_bagging <- confusionMatrix(bagging_predictions, test$Target, positive = "Liver_Disease")
cat("Bagging Model Evaluation:\n")
print(confusion_matrix_bagging)

# Generate predictions for ROC analysis
bagging_probabilities <- predict(bagging_model, test, type = "prob")[, 2]
bag_ROCR_prediction <- prediction(bagging_probabilities, test$Target)
bag_ROCR_performance <- performance(bag_ROCR_prediction, "tpr", "fpr")

# Plot the ROC curve
plot(bag_ROCR_performance, colorize = TRUE, main = "ROC Curve for Bagging Model")
abline(a = 0, b = 1, col = "red", lty = 2)

# Compute AUC
AUC_BAG <- performance(bag_ROCR_prediction, "auc")@y.values[[1]]
cat("AUC for Bagging Model:", AUC_BAG, "\n")

# ----------------------------------------------------------
# SECTION 9: RANDOM FOREST AND CROSS VALIDATION
# ----------------------------------------------------------

set.seed(5555)

# Cross-validation settings
RF_CTRL <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

# Random Forest with 10-fold CV
Model_RF <- train(
  Target ~ .,
  data = train,
  method = "ranger",
  metric = "ROC",
  trControl = RF_CTRL,
  importance = "impurity"
)

# Print model summary
print(Model_RF)

# Variable importance plot
plot(varImp(Model_RF), main = "Variable Importance - Random Forest (CV)")

# Predict probabilities for ROC
RF_probabilities <- predict(Model_RF, test, type = "prob")[, 2]
RF_ROCR_prediction <- prediction(RF_probabilities, test$Target)
RF_ROCR_performance <- performance(RF_ROCR_prediction, "tpr", "fpr")

# Plot ROC curve
plot(RF_ROCR_performance, colorize = TRUE, main = "ROC Curve - Random Forest (CV)")
abline(a = 0, b = 1, col = "green", lty = 2)

# Compute AUC
AUC_RF <- performance(RF_ROCR_prediction, "auc")@y.values[[1]]
cat("AUC for Random Forest (CV):", AUC_RF, "\n")

# Confusion matrix
RF_predictions <- predict(Model_RF, test)
confusion_matrix_RF <- confusionMatrix(RF_predictions, test$Target, positive = "Liver_Disease")

cat("Confusion Matrix - Random Forest (CV):\n")
print(confusion_matrix_RF)


# ----------------------------------------------------------
# SECTION 10: MODEL RESULTS AND INTERPRETATION
# ----------------------------------------------------------

# Print AUC Values for Comparison
cat("Final Model AUC Results\n")
cat("Decision Tree AUC:", AUC_DT, "\n")
cat("Bagging AUC:", AUC_BAG, "\n")
cat("Random Forest (CV) AUC:", AUC_RF, "\n")

# Model Comparison and Determining the best model
cat("\nModel Comparison\n")

if (AUC_RF > AUC_BAG & AUC_RF > AUC_DT) {
  cat("The model with the highest AUC is: Random Forest (CV)\n")
} else if (AUC_BAG > AUC_RF & AUC_BAG > AUC_DT) {
  cat("The model with the highest AUC is: Bagging\n")
} else {
  cat("The model with the highest AUC is: Decision Tree\n")
}

# ------ Interpretation ------
cat("\n------ Interpretation ------\n")
cat("The Decision Tree model provides a simple, interpretable baseline but tends to overfit.\n")
cat("Bagging improves performance by reducing variance through ensemble learning.\n")
cat("Random Forest with cross-validation offers the most robust and generalizable performance, achieving the highest AUC.\n")
cat("Therefore, Random Forest with cross-validation is recommended for deployment as it balances predictive accuracy and model stability.\n")

