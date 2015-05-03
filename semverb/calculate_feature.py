
#helper methods
def create_vector(file):
    lines = [line.strip().split("\t") for line in open(file, "r")]
    return lines


lines = create_vector("output")
import pdb; pdb.set_trace()
#[line if "carve" in line for line in lines]
