import numpy as np # linear algebra
import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pymongo
from pymongo import MongoClient
import datetime



print 'loading data'
orders = pd.read_csv("data/orders_train_test.csv")
prior = pd.read_csv("data/order_products__prior.csv")
train = pd.read_csv("data/order_products__train.csv")
frames = [prior, train]
products = result = pd.concat(frames)
orders = orders.loc[orders['group_indicator'] == 1]
test_orders = pd.merge(orders, products, on='order_id')
test_orders2 = test_orders[['user_id', 'order_id', 'product_id']]
test_orders2 =  test_orders2.groupby(['user_id', 'order_id'])['product_id'].apply(list)
test_set = pd.read_csv("data/test_set_.csv", names = ["user_id", "days_since_prior_order", "basket", "order_id"])
preds = pd.read_csv("data/pred-actual.csv",  names = ["pred", "actual"])
user_prod_stats = pd.read_csv("data/user_product_stats.csv")
print 'getting svd matrix'
test_preds = pd.concat([test_set, preds], axis=1)
pred_prods =pd.DataFrame()
l=int(len(test_set))
c=int(1)
final_pred_prods = []
final_pred_prods2 = pd.DataFrame()

def get_prediction(user_name):

    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client.retail_db
    users = db.users
    use = list(users.find({"text": user_name}).sort([("date", pymongo.DESCENDING)]).limit(1))
    user_name = use[0]['text']
    ub = client.user_table
    unames = ub.users
    user_list = list(unames.find({"user_name": user_name}).limit(1))
    user_id = user_list[0]['user_id']
    user_stats = []
    basket_size = int(round(test_preds[test_preds.user_id == user_id]['pred']))
    user_stats = user_prod_stats.loc[user_prod_stats['user_id'] == user_id]
    user_products = user_stats['product_id']
    pred_prods  =  user_products.head(basket_size)
    df_row = pred_prods.tolist()
    sample_user = user_name
    sample_size = basket_size
    sample_reco = pred_prods
    final_pred_prods.append(df_row)
    print 'User:', sample_user
    print 'Basket size:', int(sample_size)
    print  'recommeded products:',list(sample_reco)
