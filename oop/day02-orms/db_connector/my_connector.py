import sqlite3
import pandas as pd

class MyDBConnector(object):

    def __init__(self, db_path=None):
        self.conn = sqlite3.Connection(db_path)
        self.cursor = self.conn.cursor()
    

    def list_tables(self):
        query = "select name from sqlite_master where type='table'"
        res = self.conn.execute(query).fetchall()
        table_names = [r[0] for r in res]
        return table_names 


    def get_table_name(self, table_name='products'):
        try:
            table_name = self.table_name
        except:
            table_name = table_name
        return table_name


    def build_select_all_query(self, table_name='products'):
        return f"select * from {table_name};"    


    def select_all(self, table_name='products'):
        """ 
        runs a 'select *' query on a given table_name

        returns
        list of tuples with results from query
        """
        table_name = self.get_table_name(table_name=table_name)
        query = self.build_select_all_query(table_name=table_name)
        res = self.cursor.execute(query).fetchall()
        return res 


    def list_column_names(self, table_name='products'):
        table_name = self.get_table_name(table_name=table_name)
        query = f"PRAGMA table_info({table_name})"
        res = self.conn.execute(query).fetchall()
        column_names = [r[1] for r in res]
        return column_names 



    # load data as a dataframe
    def load_table_as_df(self, table_name='products'):
        table_name = self.get_table_name(table_name=table_name)
        query = self.build_select_all_query(table_name=table_name)
        df = pd.read_sql(query, self.conn)
        return df

    def load_query_as_df(self, query):
        df = pd.read_sql(query, self.conn)
        return df



# building a child class
class OrderDetailsConnector(MyDBConnector):

    def __init__(self, db_path=None):
        super().__init__(db_path=db_path)
        self.table_name = "orderdetails"



    def get_order_numbers(self, order_numbers):
        query = f"select * from {self.table_name} where orderNumber in {tuple(order_numbers)}"
        df = self.load_query_as_df(query=query)
        return df 


class OrdersConnector(MyDBConnector):
    
    def __init__(self, db_path=None):
        super().__init__(db_path=db_path)
        self.table_name="orders"
