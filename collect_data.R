# Possible List to collect 
# [basketball, football, tennis, volleyball, badminton, baseball, hockey, boxing, cycling, golf] 

library(twitteR)
# 
# consumer_key <- "[YOUR KEY HERE]"
# consumer_secret <- "[YOUR KEY HERE]"
# access_token <- "[YOUR KEY HERE]"
# access_secret <- "[YOUR KEY HERE]"

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)


collect_data <- function (tag, n_collect) {
    search_term <- paste('#', tag, sep='')
    cat("Collecting", n_collect, "tweets for", search_term , "...\n")
    data <- searchTwitteR(search_term, n=n_collect, lang='en')
    cat("Done collecting, converting data to dataframe ...\n")
    df <- twListToDF(data)
    file_name <- paste('~/Desktop/twitter_topic_analysis/files/data/', tag, sep='')
    cat("Saving file in csv ...\n")
    write.csv(df, file=paste(file_name, '.csv', sep=''))
    cat("Done!\n")
}

collect_data("hockey", 10000)