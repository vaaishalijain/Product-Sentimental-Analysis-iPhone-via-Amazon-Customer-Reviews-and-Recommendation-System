install.packages("RTextTools")
library(RTextTools)
#load data
SVM <- naive

#dtm
dtMatrix <- create_matrix(SVM$text)
dtMatrix

# Configure the training data
container <- create_container(dtMatrix, SVM$class, trainSize=1:150, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)
predictionData <- SVM$text[151:300]

# create a prediction document term matrix
predMatrix <- create_matrix(predictionData, originalMatrix=dtMatrix)

# create the corresponding container
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)

results <- classify_model(predictionContainer, model)
results

# Prepare the confusion matrix

conf <- confusionMatrix(results$SVM_LABEL, SVM$class[151:300])
conf
conf$overall['Accuracy']
