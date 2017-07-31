def find_start(fname, str):
    f = open(fname)
    for line in f:
        if line.startswith(str):
            print(line)


def find_in(fname, start, end):
    f = open(fname)
    for line in f:
        if line.startswith(start) and line[:-1].endswith(end):
            print(line)

# find_in("imooc.txt", "imooc", "imooc")
