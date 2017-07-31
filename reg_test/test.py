import re

line1 = "XXX出生于2001年6月1日"
line2 = "XXX出生于2001/6/1"
line3 = "XXX出生于2001-6-1"
line4 = "XXX出生于2001-06-01"
line5 = "XXX出生于2001-06"
line6 = "XXX出生于2001年6月"
regex_str = ".*?出生于(\d{4}[年/-]\d{1,2}([月/-]\d{1,2}|月|)(日|))"
match_obj = re.match(regex_str, line1)
if match_obj:
    print(match_obj.group(1))
