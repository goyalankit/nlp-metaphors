from nltk.tokenize import sent_tokenize
import re

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

file = '/Users/ankit/code/nlp-metaphors/data/subtask5b_en_allwords_test.txt'
corpus_file = open (file, "r")
output_file = open ("/Users/ankit/code/nlp-metaphors/data/semverb/just_sentences_with_phrases_test.txt", "w")

for line in corpus_file:
    line_parts = line.split("\t")
    assert len(line_parts) == 4
    try:
        sentences = sent_tokenize(line_parts[3].decode("utf8"))
        sane_sen = [sentence for sentence in sentences if "<b>" in sentence]
        sane_sen_w = remove_tags(sane_sen[0].encode("utf8"))
    except:
        import pdb; pdb.set_trace()
    final_sentence = "%s\n" % sane_sen_w
    output_file.write(final_sentence)


