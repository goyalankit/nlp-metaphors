from nltk.corpus import verbnet
from nltk.stem.porter import PorterStemmer

_PERSON_SYNSET = 'person.n.01'
_MALE_SYNSETS = [
  'male.n.02',
  'man.n.01',
  'guy.n.01',
  'chap.n.01',
]
_FEMALE_SYNSETS = [
  'female.n.02',
  'woman.n.01',
  'girl.n.01',
  'girl.n.05',
  'lady.n.01',
]
_RESTRICTION_SYNSETS = {
  'abstract': ( ['abstraction.n.06'],
                ['physical_entity.n.01']),
  'animal': ( ['animal.n.01'],
              ['person.n.01',
               'natural_object.n.01',
               'item.n.03',
               'assembly.n.05',
               'artifact.n.01']),
  'animate': ( ['living_thing.n.01'],
               ['abstraction.n.06',
                'natural_object.n.01',
                'item.n.03',
                'assembly.n.05',
                'artifact.n.01']),
  'body_part': ( ['body_part.n.01'],
                 ['abstraction.n.06',
                  'living_thing.n.01']),
  'comestible': ( ['food.n.01'],
                  ['abstraction.n.06',
                   'person.n.01',
                   'artifact.n.01']),
  'communication': ( ['communication.n.02'],
                     ['physical_entity.n.01']),
  'concrete': ( ['physical_entity.n.01'],
                ['abstraction.n.06']),
  'currency': ( ['monetary_unit.n.01'],
                ['physical_entity.n.01']),
  'elongated': ( [],
                 ['abstraction.n.06',
                  'living_thing.n.01']),
  'force': ( ['force.n.02'],
             ['living_thing.n.01']),
  'garment': ( ['garment.n.01'],
               ['abstraction.n.06',
                'living_thing.n.01']),
  'human': ( ['person.n.01'],
             ['abstraction.n.06',
              'animal.n.01',
              'natural_object.n.01',
              'item.n.03',
              'assembly.n.05',
              'artifact.n.01']),
  'int_control': ( ['living_thing.n.01',
                    'instrumentality.n.03',
                    'force.n.02'],
                   []),
  'machine': ( ['instrumentality.n.03'],
               ['abstraction.n.06',
                'living_thing.n.01']),
  'nonrigid': ( [],
                ['abstraction.n.06',
                 'living_thing.n.01']),
  'organization': ( ['organization.n.01'],
                    ['physical_entity.n.01',
                     'communication.n.02',
                     'otherworld.n.01',
                     'psychological_feature.n.01',
                     'attribute.n.02',
                     'set.n.02',
                     'measure.n.02']),
  'pointy': ( [],
              ['abstraction.n.06',
               'living_thing.n.01']),
  'shape': ( [],
             ['abstraction.n.06',
              'living_thing.n.01']),
  'solid': ( [],
             ['abstraction.n.06',
              'living_thing.n.01']),
  'sound': ( ['auditory_communication.n.01',
              'sound.n.04'],
             ['physical_entity.n.01',
              'otherworld.n.01',
              'group.n.01',
              'attribute.n.02',
              'set.n.02',
              'measure.n.02']),
  'state': ( ['state.n.02'],
             ['physical_entity.n.01']),
  'substance': ( ['substance.n.01'],
                 ['abstraction.n.06',
                  'living_thing.n.01']),
  'time': ( ['time_period.n.01',
             'clock_time.n.01'],
            ['physical_entity.n.01',
             'otherworld.n.01',
             'group.n.01',
             'attribute.n.02',
             'set.n.02',
             'communication.n.02']),
  'vehicle': ( ['vehicle.n.01'],
               ['abstraction.n.06',
                'living_thing.n.01'])
}

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

