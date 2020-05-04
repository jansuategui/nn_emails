from random import randint
from keras.preprocessing.sequence import pad_sequences


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

def predict_sequences(model, tokenizer, sequences, n_cases, n_words):
    max_seq = max([len(seq) for seq in map(lambda s: s.split(' '),sequences)]) - 1
    for i in range(n_cases):
        seed_text = sequences[randint(0,len(sequences))]
        split_text = seed_text.split(' ')
        # have at least 3 words to predict the following n_words
        while len(split_text) < 3 + n_words:
            seed_text = sequences[randint(0,len(sequences))]
            split_text = seed_text.split(' ')
        print(seed_text)
        seed_text = ' '.join(split_text[:-n_words])
        generated = generate_seq(model, tokenizer, max_seq, seed_text, n_words)
        print('Predicted last ' + str(n_words) + ' words: ' + generated)

def predict_text(model, tokenizer, text, max_seq, n_words):
    generated = generate_seq(model, tokenizer, max_seq, text, n_words)
    print(generated)
    return generated