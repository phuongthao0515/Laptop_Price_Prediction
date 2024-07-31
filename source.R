# -------------------- IMPORT LIBRARIES AND SETUP WORKSPACE --------------------
library(corrplot)
library(caTools)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(caret)
library(lattice)
library(glmnet)
library(rpart)
library(Metrics)
library(readxl)
library(car)
library(boot)
library(e1071)
library(gbm)
library(stats)
library(lmtest)

theme_set(theme_pubr())
set.seed(123)

# ---------------------------- IMPORT DATA -------------------------------------

data <- read.csv("laptops.csv")

# ------------------------- DATA PREPROCESSING ---------------------------------

# Check missing values
sum(is.na(data))
colSums(is.na(data))

# Merge two columns
data$Model <- paste(data$Brand, data$Model, sep = " ")

# Find the null value in 2 columns "GPU"
result <- data %>% filter(data$GPU == "")

GPU_frequency <- table(data$GPU)
#print(GPU_frequency)
most_frequent_value <- names(GPU_frequency)[which.max(GPU_frequency)]
#print(most_frequent_value)

result <- data %>% filter(data$Storage.type == "")
#print(result)
St_frequency <- table(data$Storage.type)
#print(St_frequency)
most_frequent_value <- names(St_frequency)[which.max(St_frequency)]
#print(most_frequent_value)

data$GPU <- ifelse(data$GPU == "", "No GPU", data$GPU)

data <- na.omit(data)
# Remove attribute "Laptop" and "Brand" and "Storage.type"
data <- data[, -which(names(data) == "Storage.type")]
data <- data[, -which(names(data) == "Laptop")]
data <- data[, -which(names(data) == "Brand")]

data <- data[data$Final.Price <= 5000, ]
variables <- setdiff(names(data), c("Final.Price"))
total_rows <- nrow(data)
for (variable in variables) {
  value_counts <- table(data[[variable]])
  values_to_remove <- names(value_counts[value_counts / total_rows <= 0.01])
  data <- data[!(data[[variable]] %in% values_to_remove), ]
}

data$CPU <- as.factor(data$CPU)
data$Status <- as.factor(data$Status)
data$Model <- as.factor(data$Model)
data$GPU <- as.factor(data$GPU)
data$Touch <- as.factor(data$Touch)
data$Screen <- as.factor(data$Screen)
data$RAM <- as.factor(data$RAM)
data$Storage <- as.factor(data$Storage)

View(data)
# ------------------------- DESCRIPTIVE STATISTICS -----------------------------

#Summary data
summary(data)
#Histogram and summary of Final Price
data %>%
  ggplot(aes(x=Final.Price)) +
  geom_histogram(binwidth = 10, fill = "deepskyblue")

summary(data$Final.Price)

#Device Status pie chart and boxplot
data_status <- as.data.frame(table(data$Status))
colnames(data_status) <- c("Status", "n")
pie(data_status$n, labels = paste(data_status$Status, scales::percent(data_status$n / sum(data_status$n))), main = "Status Distribution")

data%>%
  ggplot(aes(y = Final.Price,x = Status))+
  geom_boxplot(color="black", fill="deepskyblue") +
  theme_bw()+
  labs(x = "Computer Status", y = "Final Price")

#Ram and Storage scatter plot
data%>% 
  ggplot(aes(RAM, Final.Price)) +
  geom_point(colour = "#2ca0c5") +
  theme_bw()

data%>%
  ggplot(aes(x=Storage, y = Final.Price))+
  geom_point(colour = "#FF9999") +
  geom_smooth(method = lm, se = FALSE) +
  theme_bw()


#Model histogram and Boxplot
ggplot(data, aes(Model)) +
  geom_bar(fill = "#0073C2FF") +
  theme_pubclean() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

data%>%
  ggplot(aes(x = Final.Price,y = Model))+
  geom_boxplot(color="red", fill="orange", alpha=0.2) +
  theme_bw()+
  labs(x = "Computer Model", y = "Final Price")

#CPU histogram and boxplot
ggplot(data, aes(CPU)) +
  geom_bar(fill = "#0073C2FF") +
  theme_pubclean() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

data%>%
  ggplot(aes(x = Final.Price,y = CPU))+
  geom_boxplot(color="black", fill="deepskyblue") +
  theme_bw()+
  labs(x = "Computer Model", y = "Final Price")

ggplot(data, aes(GPU)) +
  geom_bar(fill = "#0073C2FF") +
  theme_pubclean() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

#GPU histogram and boxplot
data%>%
  ggplot(aes(x = Final.Price,y = GPU))+
  geom_boxplot(color="black", fill="deepskyblue") +
  theme_bw()+
  labs(x = "Computer Model", y = "Final Price")

#Screen size scatter plot
data%>%
  ggplot(aes(x = Screen,y = Final.Price,colour = Touch))+
  geom_point()+
  geom_smooth(method = lm,se = FALSE)+
  facet_wrap(~Touch)+
  theme_bw()
# ------------------------- INFERENTIAL STATISTICS -----------------------------

# ---------------------- MULTILINEAR REGRESSION (MLR) --------------------------

# 8 variables: Status, Model, CPU, GPU, RAM, Storage, Screen, Touch
# First, we try with full variables
#sink("split_train_summary_output.txt")
cat("-------------- Multi-linear regression with full predictors -------------\n")
train_indices_full <- createDataPartition(data$Final.Price, p = 0.7, list = FALSE)
train_data_full <- data[train_indices_full, ]
test_data_full <- data[-train_indices_full, ]
# must ensure no new data on the testing set
# all categorical variables và đã đc factor() -> sẽ báo lỗi
# riêng final.price là numerical -> ko báo lỗi

# Train the multilinear regression model
MLR_full <- lm(Final.Price ~ . - Final.Price , data = train_data_full)

# In ra summary của model
print(summary(MLR_full))
#sink()

#sink("predict_output.txt")
# Make predictions on the testing data
MLR_train_predictions <- predict(MLR_full, train_data_full)     # tự predict trên training data
MLR_test_predictions <- predict(MLR_full, test_data_full)   
accuracy_MLR_full <- data.frame(Actual = test_data_full$Final.Price, Predicted = MLR_test_predictions)  # tổng hợp 2 cột lại
#cat("The result of MLR on testing set:\n")
#print(accuracy_MLR)

MAPE_MLR_full <- mape(accuracy_MLR_full$Actual, accuracy_MLR_full$Predicted)    # dựa trên test data

cat("MAPE:", MAPE_MLR_full, "\n")
cat("train_rmse = ", summary(MLR_full)$sigma, "\ntest_rmse = ", rmse(accuracy_MLR_full$Actual, accuracy_MLR_full$Predicted))
#sink()

# ANOVA One-way TEST for MLR: any predictor with p-value >= 0.01 will be removed
#sink("anova_output.txt")
cat("\n ---------------------- Performing ANOVA test ---------------------------\n")
anova_MLR_full <- anova(MLR_full)       # this ANOVA is different from ANOVA in school
print(anova_MLR_full)
#sink()

train_data <- train_data_full
test_data <- test_data_full

#sink("VIF_output.txt")
cat(" ------------- Performing multicollinearity test with VIF ---------------\n")
vif_values <- vif(MLR_full)   # No attributes are removed
print(vif_values)
#sink()

png('MLR%03d.png', width=12, height=12, units='in', res=300)
plot(MLR_full, ask = FALSE)
dev.off()

#sink("test_output.txt")
# Independence Test (Autocorrelation Test)
print(durbinWatsonTest(MLR))
#sink()

#sink("function_output.txt")
# The final linear function
coefficients <- summary(MLR_full)$coefficients[, 1]

# Write out the linear function
equation <- "y = "
equation <- paste0(equation, round(coefficients[1], 3))     # concat coefficient đầu tiên (cũng chính là intercept)
for(i in 2:length(coefficients)) {
  # Add the coefficient and variable name to the equation
  variable_name <- paste0("x", i - 1)  # Generating x1, x2, x3, ...
  
  # This if-else condition is only used for formatting
  if (coefficients[i] >= 0) equation <- paste0(equation, " + ", round(coefficients[i], 3), "*", variable_name)
  else equation <- paste0(equation, " - ", round(abs(coefficients[i]), 3), "*", variable_name)
}
print(equation)
#sink()

head(accuracy_MLR_full, 5)


# ---------- Weighted Regression and Response Variable Log-Transformation ----------
#sink("weighted_log_train_output.txt")
cat("\n------ Weighted Regression and Response Variable Log-Transformation ------\n")
train_data_new <- train_data
train_data_new$Final.Price <- logb(train_data_new$Final.Price, exp(1)) # log-transforming the response variable (phép lấy ln())
MLR_new <- lm(Final.Price ~ . , data = train_data_new)

test_data_new <- test_data
test_data_new$Final.Price <- logb(test_data_new$Final.Price, exp(1))

wt <- 1 / lm(abs(MLR_new$residuals) ~ MLR_new$fitted.values)$fitted.values^2  # lower variance have bigger weight and vice versa
wls_model <- lm(Final.Price ~ . , data = train_data_new, weights=wt)
print(summary(wls_model))
#sink()

#sink("weighted_log_predict_output.txt")
train_predictions <- predict(wls_model, newdata = train_data_new)
new_MLR_predictions <- predict(wls_model, newdata = test_data_new)
accuracy_weightedMLR <- data.frame(Actual = test_data_new$Final.Price, Predicted = new_MLR_predictions)
MAPE_weightedMLR <- mape(exp(accuracy_weightedMLR$Actual), exp(accuracy_weightedMLR$Predicted))

cat("MAPE:", MAPE_weightedMLR, "\n")
cat("train_rmse = ", rmse(exp(train_predictions), exp(train_data_new$Final.Price)),
    "\ntest_rmse = ", rmse(exp(new_MLR_predictions), exp(test_data_new$Final.Price)), "\n")
#sink()

png('WeightedMLR%03d.png', width=12, height=12, units='in', res=300)
plot(wls_model, main = "Weighted Multi-linear Regression Model", ask = FALSE)
dev.off()

# Transformation of Categorical Variables to Dummy Variables, since cv.glmnet() does not support one-hot encoding
train_data_transformed <- model.matrix(Final.Price ~ . - 1, data = train_data)
test_data_transformed <- model.matrix(Final.Price ~ . - 1, data = test_data)

X_train_transformed <- as.matrix(train_data_transformed)
X_test_transformed <- as.matrix(test_data_transformed)
y_train <- train_data$Final.Price
y_test <- test_data$Final.Price


# ------------------------- Decision Tree Model --------------------------------
#sink("decision_tree_output.txt")
cat("\n ---------------------------- Decision Tree ----------------------------\n")
train_transformed_inc <- cbind(as.data.frame(X_train_transformed), Final.Price = y_train)
test_transformed_inc <- cbind(as.data.frame(X_test_transformed), Final.Price = y_test)
tree_model <- rpart(Final.Price ~ ., data = train_transformed_inc)
tree_predictions <- predict(tree_model, newdata = test_transformed_inc)
tree_accuracy <- data.frame(Actual = y_test, Predicted = tree_predictions)
tree_R2 <- R2(tree_predictions, y_test)
tree_MAPE <- mape(y_test, tree_predictions)
cat("MAPE:", tree_MAPE, "\n")
cat("Coefficient of Determination:", tree_R2, "\n")
print(tree_accuracy)
#sink()

# ------------------------------ Gradient Boosting -----------------------------
#sink("gradient_boosting.txt")
cat("\n ------------------------- Gradient Boosting ---------------------------\n")
# set up the tuning grid
parameter_grid <- expand.grid(n.trees = 100, interaction.depth = c(5,10), shrinkage = c(0.002, 0.02), n.minobsinnode = 10)
# set up train control
fitControl <- trainControl(method = "cv", number = 5, returnData = TRUE, verboseIter = FALSE)

gbm_model <- train(Final.Price ~ ., data = train_data, method = "gbm",
                   trControl = fitControl, verbose = FALSE, tuneGrid = parameter_grid, preProc = "zv")

gbm_predictions <- predict(gbm_model, newdata = test_data)
gbm_accuracy <- data.frame(Actual = y_test, Predicted = gbm_predictions)
gbm_R2 <- R2(gbm_predictions, y_test)
gbm_MAPE <- mape(y_test, gbm_predictions)
cat("MAPE:", gbm_MAPE, "\n")
cat("Coefficient of Determination:", gbm_R2, "\n")

#sink()
