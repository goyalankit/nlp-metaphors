import os
import subprocess

#helper methods
def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def get_srl(sentence, srl_test):
    params = "sentence=%s" % sentence.replace('"','')
    command = 'curl -X POST http://barbar.cs.lth.se:8081/parse -d "%s"' % params
    output = subprocess.check_output(command, shell=True)
    srl_test.write(output)
    srl_test.write("\n================\n")
    return

def create_file():
    test_data   = create_vector("../data/bow/train.txt")
    srl_test  = open("srl_train.txt", "a")
    count = 0
    for line in test_data:
        count += 1
        output = get_srl(line, srl_test)

    print "Finishing the script..."
    print count

create_file()


