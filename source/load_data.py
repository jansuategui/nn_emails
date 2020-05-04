import pandas as pd
import string
import nltk
from email import message_from_string
from nltk.corpus import state_union
from nltk.tokenize import sent_tokenize


def load_data():
    filepath = '../data/emails.csv'
    df = pd.read_csv(filepath)
    split_body(df)
    return df

    
def get_text_from_email(msg):
    # Get content of email using the email walk function
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    # Split email addresses when there's more than 1
    if line:
        addrs = line.split(',')
        addrs = list(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

def clean_body(body):
    return list(map(clean_sentence, sent_tokenize(body)))

def clean_sentence(sentence):
        # replace '--' with a space ' '
        body = sentence.replace('--', ' ')
        # split into tokens by white space
        tokens = sentence.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # make lower case
        tokens = [word.lower() for word in tokens]
        return tokens    

def split_body(df):
    messages = list(map(message_from_string, df['message']))
    df.drop('message', axis=1, inplace=True)
    df.drop('file', axis=1, inplace=True)
    dates = []
    senders = []
    receivers = []
    bodies = []
    for email in messages:
        dates.append(email['Date'][:-6])
        senders.append(email['From'])
        receivers.append(split_email_addresses(email['To']))
        bodies.append((get_text_from_email(email)))
    df['Date'] = pd.to_datetime(dates)
    df['From'] = senders
    df['To'] = receivers
    df['content'] = bodies
    

def write_sequence_file(sequences, filename):
    f = open(filename, 'a')
    for content in sequences:
        for seq in content:
            if len(seq) > 1:
                f.write(' '.join(seq))
                f.write('\n')

def load_sequences(filename):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    return text.split('\n')