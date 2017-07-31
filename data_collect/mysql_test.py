import pymysql.cursors

conn = pymysql.connect(host="192.168.199.171",
                       user="root",
                       password="123456",
                       db="test",
                       charset="utf8")
cur = conn.cursor()

try:
    sql = "select * from user"
    cur.execute(sql)
    conn.commit()
    for r in cur:
        print("row_number:", (cur.rownumber))
        print("id:" + str(r[0]) + " name:" + str(r[1]) + " age:" + str(r[2]))
finally:
    cur.close()
    conn.close()
