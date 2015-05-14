from __future__ import division
from nltk.corpus import verbnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk.corpus as nlcor
import nltk

#helper methods
def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

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
  'animate': ( ['living_thing.n.01', 'biological_group.n.01'],
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
        if (akey.lower() in ['i', 'he', 'she', 'we', 'you']):
            return _RESTRICTION_SYNSETS['animate']
#        elif (akey.lower() in ['it']):
#            return _RESTRICTION_SYNSETS['machine']
    #print "calling wordnet on %s" % akey
    wanalysis = WordnetAnalysis(akey.decode('utf8'))
    #print wanalysis
    return [wa.name() for wa in wanalysis]

def patient_class(patients):
    pkey = patients.keys()[0]
    pvalue = patients.values()[0]
    if (pvalue == "PRP"):
        if (pkey.lower() in ['me', 'i', 'us', 'you', 'themselves', 'him', 'her']):
            return _RESTRICTION_SYNSETS['animate']
#        elif (pkey.lower() in ['it']):
#            return _RESTRICTION_SYNSETS['machine']

    wanalysis = WordnetAnalysis(pkey.decode('utf8'))
    return [wa.name() for wa in wanalysis]

# ['Location', 'Patient1', 'Material', 'Patient', 'Source', 'Attribute', 'Destination', 'Actor2', 'Agent', 'Beneficiary', 'Instrument', 'Theme', 'Patient2', 'Experiencer', 'Actor1', 'Recipient', 'Actor', 'Asset']

all_keys = []
def check_validity(current_srl, vindex, restrictions):
    agents, patients = getAgents(current_srl, vindex)
    actors = set(["Actor2", "Agent", "Actor", "Actor1", "Actor2"]).intersection(set(restrictions.keys()))
    recipients = set(["Patient1", "Patient2", "Experiencer", "Recipient"]).intersection(set(restrictions.keys()))
    score = 3
    #agent_satisfy = True
    #patient_satisfy = True
    #include_agent = False
    #include_patient = False
    a_satisfy = False
    if agents:
        if (len(actors)!=0):
            for actor in actors:
                restriction_op = restrictions[actor][0]
                neg_false = 0
                a_satisfy = False
                changed = False

                for arestrict in restrictions[actor][1]:
                    rest = arestrict
                    if (rest[0] == '+'):
                        positive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                        word_r = agent_class(agents)
                        if not word_r:
                            continue
                        nested = any(isinstance(i, list) for i in word_r)
                        if nested:
                            result = [True for wr in word_r[0] if wr in positive_r]
                        else:
                            result = [True for wr in word_r if wr in positive_r]

                        if len(result) != 0:
                            a_satisfy = True
                            changed = True
                        else:
                            changed = True


                    elif (rest[0] == '-'):
                        if neg_false == 0:
                            a_satisfy = True

                        negatiive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                        word_r = agent_class(agents)
                        if not word_r:
                            continue

                        nested = any(isinstance(i, list) for i in word_r)
                        if nested:
                            result = [True for wr in word_r[0] if wr in negative_r]
                        else:
                            result = [True for wr in word_r if wr in negative_r]

                        if len(result) != 0:
                            neg_false = 1
                            a_satisfy = False
                            changed = True
                        else:
                            changed = True

                if (a_satisfy and (neg_false == 0 or neg_false == 1) and (changed == True)):
                    score += 1
                elif (((not a_satisfy) and (changed == True))):
                    score -= 1

        p_satisfy = False
        if patients:
            if (len(recipients)!=0):
                for recipient in recipients:
                    restriction_op = restrictions[recipient][0]

                    p_neg_false = 0
                    p_satisfy = False
                    changed = False

                    for arestrict in restrictions[recipient][1]:
                        rest = arestrict[1]
                        if (rest[0] == '+'):
                            positive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                            word_r = patient_class(patients)
                            if not word_r:
                                continue
                            nested = any(isinstance(i, list) for i in word_r)
                            if nested:
                                result = [True for wr in word_r[0] if wr in positive_r]
                            else:
                                result = [True for wr in word_r if wr in positive_r]

                            if len(result) != 0:
                                p_satisfy = True
                                changed = True
                            else:
                                changed = True

                        elif (rest[0] == '-'):
                            if (p_neg_false == 0):
                                a_satisfy = True

                            negatiive_r = _RESTRICTION_SYNSETS[rest[1]][0]
                            word_r = patient_class(patients)
                            if not word_r:
                                continue
                            nested = any(isinstance(i, list) for i in word_r)
                            if nested:
                                result = [True for wr in word_r[0] if wr in negative_r]
                            else:
                                result = [True for wr in word_r if wr in negative_r]

                            if len(result) != 0:
                                p_neg_false == 1
                                p_satisfy = False
                                changed = True
                            else:
                                changed = True


                    if (p_satisfy and (p_neg_false == 0 or p_neg_false == 1) and (changed == True)):
                        score += 1
                    elif (((not p_satisfy) and (changed == True))):
                        score -= 1

    #print restrictions
    #print current_srl
    #print "%s - %s" % (agents, patients)
    #print score
    return score


def getAgents(current_srl, vindex):
    agents = {}
    patients = {}
    found_agent = False
    found_patient = False
    for i in xrange(vindex, 0, -1):
        if current_srl[i].find("A0") != -1:
            if found_agent:
                continue
            scurr_srl = current_srl[i].split('\t')
            agents[scurr_srl[1]] = scurr_srl[4]
            found_agent = True
            if (found_agent & found_patient):
                return agents, patients
        elif current_srl[i].find("A1") != -1:
            if found_patient:
                continue
            scurr_srl = current_srl[i].split('\t')
            patients[scurr_srl[1]] = scurr_srl[4]
            found_patient = True
            if (found_agent & found_patient):
                return agents, patients
        else:
            pass
    return agents, patients


def process_srl(srl_output, actual_data, just_phrases):
    porter_stemmer = PorterStemmer()
    wn_lem = WordNetLemmatizer()
    file_open = open (srl_output, "r")
    output    = file_open.read()
    srl_output = output.split("\n================\n")
    srl_list = []
    [srl_list.append(line.strip()) for line in srl_output]

    phrase_sentence = create_vector(just_phrases)

    corpus_data = create_vector(actual_data)
    number = 0
    for line in corpus_data:
        sline       = line.split("\t")
        sense       = sline[2] # figurative or literal
        metaphor    = sline[1] # along the line <- the metaphor itself
        try:
            current_srl = srl_list[number].split("\n") # semantic role labeling of give sentece
        except:
            import pdb; pdb.set_trace()

        #mtokens = metaphor.split(" ")
        mtokens_t = word_tokenize(phrase_sentence[number])
        mtokens_t = [w for w in mtokens_t if not w.decode('utf8') in nlcor.stopwords.words('english')]
        mtokens   = filter(lambda word: word not in ",-'", mtokens_t)
        sane_mt = [mt.decode('utf8') for mt in mtokens]
        pos_mtokens = nltk.pos_tag(sane_mt)
        only_verbs = [tkn[0] for tkn in pos_mtokens if 'VB' in tkn[1]]
        #print "==============================================="
        line_score = 0
        token_count = 1
        number += 1
        #print "phrase tokens: %s" % mtokens_t
        #print "only verbs: %s" % only_verbs

        for mtoken in only_verbs:
            vnclasses = verbnet.classids(mtoken)
            if not vnclasses:
                vnclasses = verbnet.classids(wn_lem.lemmatize(mtoken))
                if not vnclasses:
                    continue
            #print "vnclasses: %s" % vnclasses

            mindex = [index for index, sl in enumerate(current_srl) if porter_stemmer.stem(mtoken) in sl.decode('utf8')]
            if not mindex:
         #       print 0
                continue
            token_count += 1

            class_score = 0
            class_count = 1
            #print '----- %s -----' % mtoken
            for vn in vnclasses:
                v=verbnet.vnclass(vn)
                try:
                    restrictions = GetVerbnetRestrictions(v)
                except:
                    continue

             #   print restrictions
                if restrictions:
                    class_score = check_validity(current_srl, mindex[0], restrictions)
                    class_count += 1
                    #print class_score
                else:
                    #print "No restrictions for %s" % vn
                    pass
            if class_count < 2:
                avg_class_score = class_score / class_count
            else:
                avg_class_score = class_score / (class_count - 1)
            #print '---------------'

            line_score += avg_class_score
            token_count += 1
        if token_count < 2:
            avg_line_score = line_score / token_count
        else:
            avg_line_score = line_score / (token_count - 1)

#        print "%s - %s - %s" % (sline[1], sline[2], line_score)
        print avg_line_score

#process_srl('srl_train.txt','../data/subtask5b_en_allwords_train.txt', '../data/semverb/just_sentences_with_phrases_train.txt')
#process_srl('srl_test.txt','../data/subtask5b_en_allwords_test.txt', '../data/semverb/just_sentences_with_phrases_test.txt')

#process_srl('../data/semverb/srl_lex_test.txt','../data/subtask5b_en_lexsample_test.txt', '../data/semverb/just_sentences_with_phrases_test_lex.txt')
#process_srl('../data/semverb/srl_lex_train.txt','../data/subtask5b_en_lexsample_train.txt', '../data/semverb/just_sentences_with_phrases_train_lex.txt')

process_srl('../data/semverb/srl_allwords_dev.txt','../data/subtask5b_en_allwords_dev.txt', '../data/semverb/just_sentences_with_phrases_dev_allwords.txt')
#process_srl('../data/semverb/srl_lex_dev.txt','../data/subtask5b_en_lexsample_dev.txt', '../data/semverb/just_sentences_with_phrases_dev_lex.txt')

