import pymysql

conn = pymysql.connect(host="192.168.199.245", port=3306, user="root", passwd="root", db="test", charset="utf8")

cur = conn.cursor()

sql = "select * from user"

cur.execute(sql)

rows = cur.fetchall()

for dr in rows:
    # print(dr)
    print("id=%s, name=%s, age=%s" % dr)

cur.close()
cur.close()
