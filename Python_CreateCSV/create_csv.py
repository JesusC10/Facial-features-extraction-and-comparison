import sys, os, natsort

def write_celeb(base_path, separator,label):
    f = open("celeb_images.csv", "w+")
    base_path += "Python_CreateCSV/"

    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in natsort.natsorted(dirnames):
            subject_path = os.path.join(dirname, surdirname)
            for filename in natsort.natsorted(os.listdir(subject_path)):
                print (subject_path)
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, label))
            label += 1
    f.close()

if __name__ == '__main__':
    base_path = sys.argv[1]
    separator = ";"
    label = 0
    write_celeb(base_path, separator,label)
