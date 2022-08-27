import pymysql

class Mysql(object):
    def __init__(self):
        try:
            self.conn = pymysql.connect(host='localhost',user='root',password='root',database='db_lapor',charset="utf8mb4")
            self.cursor = self.conn.cursor()  # Method used to get python to execute Mysql command (cursor operation)
            print("Successfully connected to database")
        except:
            print("connection failed")

    def getItems(self):
        sql= "select *from pengaduans"    #Get the contents of the food data table
        self.cursor.execute(sql)
        items = self.cursor.fetchall()  #Receive all return result lines
        return items