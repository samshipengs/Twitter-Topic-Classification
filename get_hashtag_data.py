'''
Load streaming twitter data with selected hashtag topics
The script will load data from stream contiously, stop running when you
think you have enough data.

https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

Feb 11 
'''

# laod libraries
import tweepy
from tweepy import OAuthHandler
import sys

if len(sys.argv) != 2:
	raise ValueError("Input hashtag topic!")
else:	
	hashtag_topic = "#" + str(sys.argv[1])

consumer_key = "9ksfPRs86QKSOSjwSwQ2AvQfy"
consumer_secret = "SuoUoM5S9ObKal0P2LxFxGR6JFw7uJy9IMgUBoby0jSJRMVEog"
access_token = "824393536305065984-1svM8e32qFzjSMjb8dUQ6QLC8MYvNRJ"
access_secret = "B2tj6Wa6suoBMEAQKleIqqscVV2DVJ6IuqHl8VgDsdT2H"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# use streaming APIs
from tweepy import Stream
from tweepy.streaming import StreamListener
 
MAX = 10000
class MyListener(StreamListener):
    def __init__(self):
        # super(MyListener, self).__init__()
        self.num_tweets = 0

    def on_data(self, data):
        try:
            self.num_tweets += 1
            if self.num_tweets <= MAX:
                fname = hashtag_topic + ".json"
                with open('./data/' + fname[1:], "a") as f:
                    f.write(data)
                    return True
            else:
                print "Done collecting {} tweets about {}".format(MAX, hashtag_topic)
                return False

        except BaseException as e:
            print("Error on_data: {}".format(str(e)))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
# get streaming data
print "Getting {} tweets for: {} ...".format(MAX, hashtag_topic)
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[hashtag_topic], languages=['en'])