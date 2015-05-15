import pickle


class Parameters(object):
    word_list  = []
    seen_accuracy = 0.0
    unseen_accuracy = 0.0

    def __init__(self, wl, aseen, aunseen):
        word_list = wl
        seen_accuracy = aseen
        unseen_accuracy = aunseen

 seen = pickle.load(open("seen_improvement.pickle", "rb"))
 unseen = pickle.load(open("unseen_improvement.pickle", "rb"))
 both = pickle.load(open("both_improvement.pickle", "rb"))
 objs = pickle.load(open("both_improvement_object.pickle", "rb"))
