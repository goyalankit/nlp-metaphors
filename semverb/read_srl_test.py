from nltk.corpus import verbnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn

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

def WordnetAnalysis(word):
    snets = wn.synsets(word)
    hyper = lambda s: s.hypernyms()
    restrictions = []
    for snet in snets:
        restrictions.extend(list(snet.closure(hyper)))
    return restrictions

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


def agent_class(agents):
    akey   = agents.keys()[0]
    avalue = agents.values()[0]
    if (avalue == "PRP"):
        if (akey.lower() in ['i', 'he', 'She', 'we', 'you']):
            return _RESTRICTION_SYNSETS['animate']
        elif (akey.lower() in ['it']):
            return _RESTRICTION_SYNSETS['machine']
    #print "calling wordnet on %s" % akey
    wanalysis = WordnetAnalysis(akey.decode('utf8'))
    #print wanalysis
    return [wa.name() for wa in wanalysis]

def patient_class(patients):
    pkey = patients.key()[0]
    pvalue = patients.values()[0]
    if (pvalue == "PRP"):
        if (pkey.lower() in ['me', 'I', 'us', 'you', 'themselves', 'him', 'her']):
            return _RESTRICTION_SYNSETS['animate']
        elif (pkey.lower() in ['it']):
            return _RESTRICTION_SYNSETS['machine']

    wanalysis = WordnetAnalysis(pkey.decode('utf8'))
    return [wa.name() for wa in wanalysis]

# ['Location', 'Patient1', 'Material', 'Patient', 'Source', 'Attribute', 'Destination', 'Actor2', 'Agent', 'Beneficiary', 'Instrument', 'Theme', 'Patient2', 'Experiencer', 'Actor1', 'Recipient', 'Actor', 'Asset']

all_keys = []
def check_validity(current_srl, vindex, restrictions):
    agents, patients = getAgents(current_srl, vindex)
    actors = set(["Actor2", "Agent", "Actor", "Actor1", "Actor2"]).intersection(set(restrictions.keys()))
    score = 4
    #('and', [('+', 'animate')])
    if agents:
        if (len(actors)!=0):
            for actor in actors:
                rest = restrictions[actor][1][0]
                if (rest[0] == '+'):
                    positive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                    word_r = agent_class(agents)
                    result = [True for wr in word_r if wr in positive_r]
                    if len(result) != 0:
                        score += 1
                    else:
                        score -= 1
                elif (rest[0] == '-'):
                    negatiive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                    word_r = agent_class(agents)
                    result = [True for wr in word_r if wr in negative_r]
                    if len(result) != 0:
                        score -= 1
                    else:
                        score += 1
        if patients:
            if (patients.values()[0] == "PRP"):
                all_keys.append(patients.keys()[0])



    print score

                #if agents and agents.values()[0] == "PRP":
                #    all_keys.append(agents.keys())


    pass


def getAgents(current_srl, vindex):
    agents = {}
    patients = {}
    found_agent = False
    found_patient = False
    for i in xrange(vindex, 0, -1):
        if current_srl[i].find("A0") != -1:
            scurr_srl = current_srl[i].split('\t')
            agents[scurr_srl[1]] = scurr_srl[4]
            found_agent = True
            if (found_agent & found_patient):
                return agents, patients
        elif current_srl[i].find("A1") != -1:
            scurr_srl = current_srl[i].split('\t')
            patients[scurr_srl[1]] = scurr_srl[4]
            found_patient = True
            if (found_agent & found_patient):
                return agents, patients
        else:
            pass
    return agents, patients


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

