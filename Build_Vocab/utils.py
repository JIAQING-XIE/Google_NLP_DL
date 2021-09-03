import json
def print_dict(ddict):
    for k,v in ddict.items():
        print("{}\t{}".format(k,v))

def dict2file(ddict, f_type = "dict"):
    f_name = "defaultdict_{}.txt".format(f_type)
    a_file = open(f_name, "w")
    if f_type == "dict":
        json.dump(ddict, a_file)
    elif f_type == "print":
        for k,v in ddict.items():
            a_file.write("{}\t\t{}\t\n".format(k,v))
    else:
        assert f_type == "dict" or f_type == "print", "pleasechoose from dict or print."
    a_file.close()


def read_dict_from_file(f_name):
    f = open(f_name, 'r')
    ddict = f.read()
    f.close()
    return ddict