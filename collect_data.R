# List to collect 
# [basketball, football, tennis, volleyball, badminton, baseball, hockey, boxing, cycling, golf] 

library(twitteR)
# 
# consumer_key <- "9ksfPRs86QKSOSjwSwQ2AvQfy"
# consumer_secret <- "SuoUoM5S9ObKal0P2LxFxGR6JFw7uJy9IMgUBoby0jSJRMVEog"
# access_token <- "824393536305065984-1svM8e32qFzjSMjb8dUQ6QLC8MYvNRJ"
# access_secret <- "B2tj6Wa6suoBMEAQKleIqqscVV2DVJ6IuqHl8VgDsdT2H"

# second keys
consumer_key <- "LlWxm5SxuseZWXNM0EIVyErU3"
consumer_secret <- "AdfjchDQJ7INCIw8aqj8DhAf8BmPW8bNFVez93uYIpEYpV07Gu"
access_token <- "2299125596-P6Z6poQHt3Kq4WHLYoGYL7I7j4VdmcC15aROACz"
access_secret <- "b7w6Xwkkqi3pHNqczTDxFWp1EuGuPFJ9UkwuX3Jau6V9X"

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