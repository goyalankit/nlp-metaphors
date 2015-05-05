from nltk.corpus import verbnet
from nltk.stem.porter import PorterStemmer

#helper methods
def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def GetVerbnetRestrictions(vnclass):
  role_restrictions = {}

  while True:
    for role in vnclass.findall('THEMROLES/THEMROLE'):
      restrictions = role.find('SELRESTRS')
      if restrictions:
        restriction_set = set()
        for restriction in restrictions.findall('SELRESTR'):
          predicate = restriction.attrib
          restriction_set.add((predicate['Value'], predicate['type']))

        total = (restrictions.get('logic', 'and'), list(restriction_set))
        role_restrictions[role.attrib['type']] = total

    if vnclass.tag == 'VNCLASS':
      break
    else:
      parent_class = vnclass.attrib['ID'].rsplit('-', 1)[0]
      vnclass = verbnet.vnclass(parent_class)

  return role_restrictions


all_keys = []
def check_validity(current_srl, vindex, restrictions):
    all_keys.extend(restrictions.keys())
    pass

# ['Location', 'Patient1', 'Material', 'Patient', 'Source', 'Attribute', 'Destination', 'Actor2', 'Agent', 'Beneficiary', 'Instrument', 'Theme', 'Patient2', 'Experiencer', 'Actor1', 'Recipient', 'Actor', 'Asset']

def process_srl(srl_output, actual_data):
    porter_stemmer = PorterStemmer()
    file_open = open (srl_output, "r")
    output    = file_open.read()
    srl_output = output.split("\n================\n")
    srl_list = []
    [srl_list.append(line.strip()) for line in srl_output]

    corpus_data = create_vector(actual_data)
    number = 0
    for line in corpus_data:
        sline       = line.split("\t")
        sense       = sline[2] # figurative or literal
        metaphor    = sline[1] # along the line <- the metaphor itself
        current_srl = srl_list[number].split("\n") # semantic role labeling of give sentece

        mtokens = metaphor.split(" ")
        for mtoken in mtokens:
            vnclasses = verbnet.classids(mtoken)
            if not vnclasses:
                continue
            mindex = [index for index, sl in enumerate(current_srl) if porter_stemmer.stem(mtoken) in sl.decode('utf8')]
            if not mindex:
                print line
                continue
            for vn in vnclasses:
                v=verbnet.vnclass(vn)
                restrictions = GetVerbnetRestrictions(v)
                check_validity(current_srl, mindex[0], restrictions)
        number += 1

process_srl('srl_test.txt','../data/subtask5b_en_allwords_test.txt')
import pdb; pdb.set_trace()

