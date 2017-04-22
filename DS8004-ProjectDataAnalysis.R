
#-------------------------------------
# Title:  DS8004 Project - Data Analysis
# Name:   Anil Trikha
# Date:   April 21, 2017
#-------------------------------------


df <- read.csv ("CollectedData/all_sports_new.csv", stringsAsFactors = FALSE)

tabulate (df$class)
levels(df$class)
unique(df$class)


library(RTextTools)
library(e1071)
library(twitteR)
library(ggplot2)

set.seed(1234)
labels <- c("basketball", "hockey", "baseball", "tennis", "volleyball")
#df <- df[df$class %in% labels,]

nPerClass = 500 # 4755

df_new <- head (df[df$class==c(1),], nPerClass)
df_new <- rbind (df_new, head (df[df$class==c(2),], nPerClass))
df_new <- rbind (df_new, head (df[df$class==c(3),], nPerClass))
df_new <- rbind (df_new, head (df[df$class==c(4),], nPerClass))
df_new <- rbind (df_new, head (df[df$class==c(5),], nPerClass))

colnames(df_new) <- c("text", "class", "tokenized")
# df<- df_new[df_new$class != c(5),]
df <- df_new
ind <- 1:length(df$class)


trainInd <- sample (ind, size = 0.7 * length(ind), replace = FALSE)

train_tweets <- cbind (df$text[trainInd], labels[df$class[trainInd]])
test_tweets <- cbind (df$text[-trainInd], labels[df$class[-trainInd]])


tweets = rbind(train_tweets, test_tweets)

# Then we can build the document-term matrix:

# build dtm
matrix= create_matrix(tweets[,1], language="english",
                      removeStopwords=TRUE, removeNumbers=TRUE, 
                      stemWords=FALSE, removeSparseTerms = 0.985) 

# Now, we can train the naive Bayes model with the training set. Note that, e1071 asks the response variable to be numeric or factor. Thus, we convert characters to factors here. This is a little trick.

dim(matrix)
trainLen <- nrow(train_tweets)
testLen <- nrow(test_tweets)

# trainInd <- sample (1:trainLen, 1000, replace=F)

# train the model
mat <- as.matrix(matrix)
classifier <- naiveBayes(mat[1:trainLen,], as.factor(tweets[1:trainLen,2]) )

# Now we can step further to test the accuracy.

# test the validity
predicted = predict(classifier, mat[(trainLen+1):(trainLen+testLen),]) #; predicted

t <- table(tweets[(trainLen+1):(trainLen+testLen), 2], predicted)

tmat <- as.matrix(t)
tmat

# tp <- tmat["positive", "positive"]
# fp <- tmat["negative", "positive"]
# tn <- tmat["negative", "negative"]
# fn <- tmat["positive", "negative"]

# precision <- tp / (tp + fp)
# precision

# recall <- tp / (tp + fn)
# recall

# accuracy <- (tp + tn) / (tp + fp + tn + fn)
accuracy <- sum(diag(tmat)) / sum(tmat)
accuracy

# recall_accuracy(tweets[(trainLen+1):(trainLen+testLen), 2], predicted)

cm <- data.frame(tweets[(trainLen+1):(trainLen+testLen), 2], predicted)
names(cm) <- c("Actual", "Predicted")

ggplot(cm, aes(x = Actual)) +
  geom_bar(fill = "midnightblue") + 
  facet_grid (. ~ Predicted, labeller=label_both) + 
  ylab("Number of Tweets") + xlab("Actual") +
  ggtitle("Tweet Prediction Results")


hist (mat[,"nba"], main = "Distribution of NBA term in Document-Term Matrix",
      xlab = "Occurrences of NBA term")

#----------------------------------
# Random Forest
#----------------------------------

library(randomForest)


forest_algo <- function (d, ind, runType)
{
  lastCol <- dim(d)[2]
  d.training <- d[ind==1, 1:lastCol-1]
  d.test <- d[ind==2, 1:lastCol-1]
  d.trainLabels <- as.factor (d[ind==1, lastCol])
  d.testLabels <- as.factor (d[ind==2, lastCol])
  results <- list()
  
  if (runType == "Iris") {
    rfFormula <- "iris_label ~ ."
  } else if (runType == "Lens") {
    rfFormula <- "lens_label ~ ."
  }
  
  model <- randomForest (d.training, d.trainLabels)
  d_pred <- predict(model, d.test) # round this value?
  
  # evaluate model
  results[[1]] <- table(d_pred, d.testLabels)
  
  results
}

ind <- rep(0, length(df$class))
ind[trainInd] <- 1
ind[-trainInd] <- 2
lens_data <- as.data.frame(cbind(mat, labels[df$class]))
colnames(lens_data)[ncol(lens_data)] <- "lens_label"
resRNF.lens <- forest_algo (lens_data, ind, "Lens")


getForestAccuracy <- function (xx) 
{ 
  x <- as.data.frame(xx)
  x$d_pred <- factor (x$d_pred, levels=c(levels (x$d_pred), 
                                         setdiff (levels(x$d.testLabels), levels(x$d_pred))))
  sum(x[which(x$d_pred == x$d.testLabels),]$Freq) / sum(x$Freq)
}

print (paste ("Random Forest: ", "Lens"))
print (paste ("Test Accuracy: ", getForestAccuracy(resRNF.lens[[1]])))
print (t(resRNF.lens[[1]]))

x <- as.data.frame(resRNF.lens[[1]])
sum(x$Freq)


#----------------------------------------------------
# SVM Algorithm applied to both datasets
#----------------------------------------------------

library(stats)

svm_kernels = c("linear", "polynomial", "radial", "sigmoid")

svm_algo <- function (d, ind, runType)
{
  lastCol <- dim(d)[2]
  d.training <- sapply(d[ind==1, 1:lastCol-1], as.numeric)
  d.test <- sapply(d[ind==2, 1:lastCol-1], as.numeric) 
  d.trainLabels <- sapply (d[ind==1, lastCol], as.factor)
  d.testLabels <- sapply (d[ind==2, lastCol], as.factor)
  results <- list()
  
  
  for (i in 1:length(svm_kernels))
  {
    model <- svm(d.training, d.trainLabels, kernel = svm_kernels[i]) 
    d_pred <- predict(model, d.test)
    
    # evaluate model
    results[[i]] <- table(d_pred, d.testLabels)
  }
  
  results
}


resSVM.lens <- svm_algo (lens_data, ind, "Lens")


getSvmAccuracy <- function (xx) 
{ 
  x <- as.data.frame(xx)
  x$d_pred <- factor (x$d_pred, levels=c(levels (x$d_pred), 
                                         setdiff (levels(x$d.testLabels), levels(x$d_pred))))
  # x$d.testLabels <- factor (x$d.testLabels, levels=c(levels (x$d.testLabels), 
  #                                        setdiff (levels(x$d_pred), levels(x$d.testLabels))))
  sum(x[which(x$d_pred == x$d.testLabels),]$Freq) / sum(x$Freq)
}



barplot (sapply (resSVM.lens, getSvmAccuracy), names.arg = svm_kernels,
         col = 1:length(svm_kernels), main = "SVM Accuracy For Sports Data", 
         xlab = "SVM Kernel", ylab = "Test Accuracy")

svm_accuracy <- function (res)
{
  for (i in 1:length(svm_kernels))
  {
    print (paste ("SVM Kernel: ", svm_kernels[i]))
    print (paste ("Test Accuracy: ", getSvmAccuracy(res[[i]])))
    print (t(res[[i]]))
  }
}

svm_accuracy (resSVM.lens)


