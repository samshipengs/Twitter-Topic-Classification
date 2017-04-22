
#-------------------------------------
# Title:  DS8004 Project - Data Analysis
# Name:   Anil Trikha
# Date:   April 8, 2017
#-------------------------------------


df <- read.csv ("~/Desktop/twitter_topic_analysis/files/data/all_sports_new.csv", stringsAsFactors = FALSE)

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
ind <- 1:length(df$class)

trainInd <- sample (ind, size = 0.7 * length(ind), replace = FALSE)

train_tweets <- cbind (df$text[trainInd], labels[df$class[trainInd]])
test_tweets <- cbind (df$text[-trainInd], labels[df$class[-trainInd]])


tweets = rbind(train_tweets, test_tweets)

# Then we can build the document-term matrix:

# build dtm
matrix= create_matrix(tweets[,1], language="english",
                      removeStopwords=TRUE, removeNumbers=TRUE, 
                      stemWords=FALSE, removeSparseTerms = 0.99) 

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

#accuracy <- (tp + tn) / (tp + fp + tn + fn)
#accuracy <- sum(diag(tmat)) / sum(tmat)
#accuracy

# compute harsh accuracy
n_class <- 5
accuracy <- 0
for (i in 1:n_class) {
    accuracy <- accuracy + tmat[i, i]/sum(tmat[i,])/n_class 
}
print(paste("Harsh accuracy is:", accuracy))
# recall_accuracy(tweets[(trainLen+1):(trainLen+testLen), 2], predicted)

# cm <- data.frame(tweets[(trainLen+1):(trainLen+testLen), 2], predicted)
# names(cm) <- c("Actual", "Predicted")
# 
# ggplot(cm, aes(x = Actual)) +
#   geom_bar(fill = "midnightblue") + 
#   facet_grid (. ~ Predicted, labeller=label_both) + 
#   ylab("Number of Tweets") + xlab("Actual") +
#   ggtitle("Tweet Prediction Results")


