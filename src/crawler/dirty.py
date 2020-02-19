import os
import time

import mysql.connector
import requests

from src.data.basic_functions import root_dir


class DataBase:
    def __init__(self):
        self.database = self.__connect_to_database()
        self.cursor = self.database.cursor(prepared=True)

    @staticmethod
    def __get_database_name_password():
        with open(os.path.join(root_dir(), "config", "data_base_password.txt"), "r") as f:
            data = f.read()
        return data.split()

    def __connect_to_database(self):
        mydb = mysql.connector.connect(
            host="localhost",
            user=self.__get_database_name_password()[0],
            passwd=self.__get_database_name_password()[1],
            auth_plugin='mysql_native_password',
            database="dirty"
        )
        return mydb

    def insert_data_to_dirty(self, data):
        sql = "INSERT INTO sentence (texts, types) VALUES (%s, %s)"
        self.cursor.execute(sql, data)
        self.database.commit()


class DirtySpider:
    """
    normal_type: 0
    """
    normal_url = "https://nmsl.shadiao.app/api.php?lang=zh_cn"
    hard_url = "https://nmsl.shadiao.app/api.php?level=min&lang=zh_cn"
    hk_url = "https://nmsl.shadiao.app/api.php?lang=zh_hk"

    def __init__(self, data_base: DataBase):
        self.data_base = data_base
        self.full_dirty = set()
        self.repeat_count = 0

    @staticmethod
    def get_page(url):
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"}
        r = requests.get(url, verify=False, proxies=None,
                         headers=headers,
                         timeout=10)
        return r.text

    def insert_data(self, data):
        if data[0] not in self.full_dirty:
            self.full_dirty.add(data[0])
            self.repeat_count = 0
            self.data_base.insert_data_to_dirty(data)
        else:
            self.repeat_count += 1

    def get_normal_dirty(self, dirty_type: [0, 1, 2]):
        """

        :param dirty_type: 0:normal, 1:hard_mode
        :return:
        """
        stop_condition = 49

        if dirty_type == 0:
            url = self.normal_url
        if dirty_type == 1:
            url = self.hard_url
        if dirty_type == 2:
            url = self.hk_url

        while self.repeat_count < stop_condition:
            data = self.get_page(url)
            self.insert_data((data, dirty_type))
            print(data)
            time.sleep(0.5)


if __name__ == '__main__':
    DirtySpider(DataBase()).get_normal_dirty(dirty_type=2)
