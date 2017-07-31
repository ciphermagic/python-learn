import sys
import pymysql


class TransferMoney(object):
    def __init__(self, conn):
        self.conn = conn

    def transfer(self, source_id, target_id, money):
        try:
            self.check_acct_available(source_id)
            self.check_acct_available(target_id)
            self.has_enough_money(source_id, money)
            self.reduce_money(source_id, money)
            self.add_money(target_id, money)
            self.conn.commit()
            print("转账成功")
        except Exception as e:
            self.conn.rollback()
            raise Exception(e)

    def check_acct_available(self, id):
        try:
            cursor = self.conn.cursor()
            sql = "select * from account where id=%s" % id
            cursor.execute(sql)
            rs = cursor.fetchall()
            if len(rs) != 1:
                raise Exception("账号%s不存在" % id)
        except Exception as e:
            self.conn.rollback()
            cursor.close()
            raise Exception(e)

    def has_enough_money(self, id, money):
        try:
            cursor = self.conn.cursor()
            sql = "select * from account where id=%s and money>=%s" % (id, money)
            cursor.execute(sql)
            rs = cursor.fetchall()
            if len(rs) != 1:
                raise Exception("账号%s没有足够的钱" % id)
        except Exception as e:
            self.conn.rollback()
            cursor.close()
            raise Exception(e)

    def reduce_money(self, id, money):
        try:
            cursor = self.conn.cursor()
            sql = "update account set money=money-%s where id=%s " % (money, id)
            cursor.execute(sql)
            if cursor.rowcount != 1:
                raise Exception("账号%s扣钱失败" % id)
        except Exception as e:
            self.conn.rollback()
            cursor.close()
            raise Exception(e)

    def add_money(self, id, money):
        try:
            cursor = self.conn.cursor()
            sql = "update account set money=money+%s where id=%s " % (money, id)
            cursor.execute(sql)
            if cursor.rowcount != 1:
                raise Exception("账号%s加钱失败" % id)
        except Exception as e:
            self.conn.rollback()
            cursor.close()
            raise Exception(e)


if __name__ == "__main__":
    source_id = input("source_id: ")
    target_id = input("target_id: ")
    money = input("money: ")

    conn = pymysql.connect(host="192.168.199.245", port=3306, user="root", passwd="root", db="test", charset="utf8")
    tr_money = TransferMoney(conn)

    try:
        tr_money.transfer(source_id, target_id, money)
    except Exception as e:
        print(e)
    finally:
        conn.close()
