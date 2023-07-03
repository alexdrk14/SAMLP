""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
MongoDB connection class
####################################################################################################################"""

import mongoConfig as cnf
from pymongo import MongoClient


class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self, getCursor=False):
        self.client = MongoClient(cnf.MONGO["ip"], port=cnf.MONGO["port"])
        self.db = self.client[cnf.MONGO["db"]]
        if getCursor:
            return self.client, self.db

    def close(self):
        self.client.close()
        self.client = None
        self.db = None

    def getUserProfile(self, user_id):
        return self.db[cnf.MONGO["collection"]].find_one({"id": int(user_id)})

