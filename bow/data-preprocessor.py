import re

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)



#datasets = {"all": { "train" :  { "main" : "subtask5b_en_allwords_train.txt", "data" : "all_train.txt", "label" : "all_label.txt" } ,"test" : {"main" : "subtask5b_en_lexsample_test.txt", "data" : "all_test.txt", "label" : "all_test_label"}}, {"lex_train" : { "main" : "subtask5b_en_lexsample_train.txt", "data" : "lex_train", "label" : "lex_label" }, "lext_test" : { "main" : "subtask5b_en_lexsample_test.txt", "data" : "lex_test.txt", "label" : "lex_text_label"} }}



def clean_data():
    corpus_data = []
    corpus_target = []
    print "Reading data..."
    corpus_file = open ("/Users/ankit/code/nlp-metaphors/data/subtask5b_en_lexsample_test.txt", "r")
    data_file   = open ("/Users/ankit/code/nlp-metaphors/data/bow/lex_test.txt", "w")
    label_file  = open ("/Users/ankit/code/nlp-metaphors/data/bow/lex_test_label.txt", "w")

    print "Cleaning data..."
    for line in corpus_file:
        line_parts = line.split("\t")
        # data validity check
        assert len(line_parts) == 4
        line_parts[3] = remove_tags(line_parts[3])
        data_file.write(line_parts[3])
        #corpus_data.append(line_parts[3])

        target_value = "0\n" if (line_parts[2] == "figuratively") else "1\n"

        label_file.write(target_value)
        #corpus_target.append(target_value)

        assert len(corpus_target) == len(corpus_data)
    print "Data cleaned..."

clean_data()


