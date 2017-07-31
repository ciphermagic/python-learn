import re

pattern = r"\w{6,9}@163.com"

str = "abc1234@163.com cipher@163.com"

try:
    print(re.match(pattern, str).group())
except Exception as e:
    print("none")

try:
    print(re.search(pattern, str).group())
except Exception as e:
    print("none")

print(re.findall(pattern, str))

str2 = "imooc:c++ c;java python"

print(re.split(r":|;| ", str2))
