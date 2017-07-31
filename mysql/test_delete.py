import pymysql

conn = pymysql.connect(host="192.168.199.245", port=3306, user="root", passwd="root", db="test", charset="utf8")
conn.autocommit(False)

cur = conn.cursor()

sql = "delete from user where name = 'test'"

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
