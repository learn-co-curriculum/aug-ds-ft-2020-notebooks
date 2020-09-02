# grab twitter data
# write twitter data to mongo collection

import json
import tweepy

from pymongo import MongoClient


def get_mongo_client(host='localhost', port=27017):
    mongo_client = MongoClient(host=host, port=port)
    return mongo_client



def get_twitter_collection(mongo_client):
    db = mongo_client['new_db']
    twitter_collection = db['tweets']
    return twitter_collection
    

def get_twitter_api():
    with open("") as f:
        d = json.load(f)
        auth = tweepy.OAuthHandler(consumer_key=d['consumer_key'], 
                                   consumer_secret=d['consumer_secret'])
    twitter_api = tweepy.API(auth)
    return twitter_api




def get_tweets(twitter_api, search="#happy"):
    tweets = []
    for res in twitter_api.search(search):
        try:
            tweets.append(res._json)
        except:
            print(f"could not find _json response for tweet:\n{res}")
    return tweets
              


    
if __name__=="__main__":
    mongo_client = get_mongo_client()
    twitter_collection = get_twitter_collection(mongo_client=mongo_client)
    original_count = twitter_collection.count_documents({})
    twitter_api = get_twitter_api()
    tweets = get_tweets(twitter_api=twitter_api)
    twitter_collection.insert_many(tweets)
    new_count = twitter_collection.count_documents({})
    print(f"started with {original_count}, ended with {new_count}, added {new_count-original_count}")

    
    