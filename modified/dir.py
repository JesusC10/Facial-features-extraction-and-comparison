import os



for i in range(1, 41):
    os.mkdir("s" + str(i))

dir = 1
i = 1

while dir <= 40 and i <= 400:
    for j in range(1, 11):
        os.rename(str(i) + ".jpg", "s" + str(dir) + "/" + str(j) + ".jpg")
        i += 1
    dir += 1
