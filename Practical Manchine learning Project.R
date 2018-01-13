
## Loading Libraries
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(corrplot)



## Loading and preprocessing the data. Treat "#DIV/0!",blank, string "NA"as missing values.
HARdatatrain <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
HARdatatest <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

## Since data after Column 8 is all numeric, Cast these column as Numeric
for(i in c(8:ncol(HARdatatrain)-1)) {HARdatatrain[,i] = as.numeric(as.character(HARdatatrain[,i]))}

for(i in c(8:ncol(HARdatatest)-1)) {HARdatatest[,i] = as.numeric(as.character(HARdatatest[,i]))}

set.seed(1234)

## Remove Nas

AllNAs    <- sapply(HARdatatrain, function(x) mean(is.na(x))) > 0.95
HARdatatrain <- HARdatatrain[, AllNAs==FALSE]
HARdatatest  <- HARdatatest [, AllNAs==FALSE]

## Remove Near zero variances
NZV <- nearZeroVar(HARdatatrain)
HARdatatrain<- HARdatatrain[, -NZV]
HARdatatest <- HARdatatest[, -NZV]

# remove identification only variables (columns 1 to 5)
HARdatatrain <- HARdatatrain[, -(1:5)]
HARdatatest <- HARdatatest[, -(1:5)]

## partition the training data 
HARinTrain <- createDataPartition(y=HARdatatrain$classe,p=0.75, list=FALSE)
HARtraining <- HARdatatrain[HARinTrain,]
HARtesting <- HARdatatrain[-HARinTrain,]

# Find the correlation between different features in Training data and plot to visualize the correlation to see
# influencing features.
corMatrix <- cor(HARtraining[, -54])
corrplot(corMatrix, order = "hclust", type = "lower", tl.srt = 45, tl.col = rgb(0, 0, 0))
corrmt1 <- findCorrelation(corMatrix, cutoff = 0.85, names = TRUE)
length(corrmt1)

## Extract features which are uncorrelated for feature plotting.
plotfeatures <- c("roll_belt", "pitch_belt", "yaw_belt", "roll_arm", "pitch_arm", 
              "yaw_arm", "roll_forearm", "pitch_forearm", "yaw_forearm", "roll_dumbbell", 
              "pitch_dumbbell", "yaw_dumbbell")
class_feature <- c("classe")
plottrainData <- HARtraining[, c(plotfeatures, class_feature)]
plottestData <- HARtesting[, plotfeatures]


## Do basic Exploratory plots
featurePlot(x = plottrainData[, plotfeatures],
            y = plottrainData[, class_feature],
            plot = "density",
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            adjust = 1.5,
            pch = "|",
            auto.key = list(columns = 3))

## Cross validation

TrainControl <- trainControl(method = "cv", number = 4) 



## fit a decision tree model first
modFit1 <- rpart(classe ~ ., data=HARtraining, method="class")
predict1 <- predict(modFit1, HARtesting, type = "class")
confusionMatrix(predict1, HARtesting$classe) 


##fit a random forest prediction model
modFit2 <- train(classe ~ ., data=HARtraining, method="rf",trControl=TrainControl)
modFit2$finalModel
predictRandForest <- predict(modFit2, newdata=HARtesting)
confusionMatrix(predictRandForest, HARtesting$classe)

## Predicitng on the test set the best model
predicTest<- predict(modFit2, HARdatatest , type = "class")
predictTest

## print to respective files
write_files = function(x){
       n = length(x)
      for(i in 1:n){
             filename = paste0("id_",i,".txt")
           write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
   }
write_files(predicTest)
