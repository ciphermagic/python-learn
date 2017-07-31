import pymysql

conn = pymysql.connect(host="192.168.199.171", port=3306, user="root", passwd="123456", db="test", charset="utf8")
conn.autocommit(False)

cur = conn.cursor()

sql = "insert into user(name, age) values('test', 33)"

try:
    cur.execute(sql)
    conn.commit()
except Exception as e:
    conn.rollback()
    print(e)

rows = cur.rowcount
print(rows)

cur.close()
cur.close()
