import re

str1 = "imooc java"

pa = re.compile(r"imooc", re.I)

ma = pa.match(str1)

print(ma.group())

print(ma.span())

print(ma.string)
