import sys
import os

if __name__ == '__main__':
    base_path = sys.argv[1]
    separator = ";"
    label = 0

    f = open("data.csv", "w+")
    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in sorted(dirnames):
            subject_path = os.path.join(dirname, surdirname)
            for filename in sorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, label))
            label += 1
    f.close()
