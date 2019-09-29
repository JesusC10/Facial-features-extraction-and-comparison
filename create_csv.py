import sys, os, natsort




def write_modified(base_path, separator,label):
    f = open("modified_data.csv", "w+")
    base_path += "modified"
    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in natsort.natsorted(dirnames):
            # print (surdirname)
            subject_path = os.path.join(dirname, surdirname)
            for filename in natsort.natsorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, label))
            label += 1
    f.close()

def write_input(base_path, separator):
    f = open("input_data.csv", "w+")
    base_path += "input_images_pgm"
    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in natsort.natsorted(dirnames):
            subject_path = os.path.join(dirname, surdirname)
            for filename in natsort.natsorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, -1))
    f.close()

if __name__ == '__main__':
    base_path = sys.argv[1]
    separator = ";"
    label = 0
    write_modified(base_path, separator,label)
    write_input(base_path, separator)
