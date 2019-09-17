import sys
import os



def write_modified(base_path, separator,label):
    f = open("modified_data.csv", "w+")
    base_path += "modified"
    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in sorted(dirnames):
            subject_path = os.path.join(dirname, surdirname)
            for filename in sorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, label))
            label += 1
    f.close()

def write_test(base_path, separator, label):
    f = open("test_data.csv", "w+")
    base_path += "test_faces"
    for dirname, dirnames, filenames in sorted(os.walk(base_path)):
        for surdirname in sorted(dirnames):
            subject_path = os.path.join(dirname, surdirname)
            for filename in sorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print(abs_path)
                f.write("%s%s%d\n" % (abs_path, separator, label))
            label += 1
    f.close()

if __name__ == '__main__':
    base_path = sys.argv[1]
    separator = ";"
    label = 0
    write_modified(base_path, separator,label)
    write_test(base_path,separator,label)
