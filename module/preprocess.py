from __future__ import unicode_literals
import re
import sys
import demoji
from PersianStemmer import PersianStemmer
from hazm import *
from transformers import AutoTokenizer
import pandas as pd


def normalize_sentence(dataframe):
    new_df = pd.DataFrame(columns=['sentence', 'label'])
    normalizer = Normalizer()
    for index in range(len(dataframe)):
        current_sentence = str(dataframe['sentence'].iloc[index])
        normalized_sentence = normalizer.normalize(current_sentence)
        new_df = new_df.append({'sentence': normalized_sentence, 'label': dataframe['label'].iloc[index]},
                               ignore_index=True)
                               
    new_df = new_df.astype({'label': 'int'})
    return new_df


def clean_text(dataframe, stop_words_type, stemming, remove_outlier=True):
    new_df = pd.DataFrame(columns=['sentence', 'label'])
    if stop_words_type == 'general':
        print('general stop words selected')
        # general stop words source: https://github.com/kharazi/persian-stopwords/blob/master/short
        stop_words = open('/content/drive/MyDrive/data/persian_short_stopwords', encoding='utf-8')

        general_stop_words = [stop_word.strip() for stop_word in stop_words]
        general_stop_words.pop()

    elif stop_words_type == 'custom':
        print('custom stop words selected')
        custom_stop_words = ['این', 'و', 'در', 'که', 'را', 'رو', 'از', 'به', 'تا', 'من', 'با', 'هم']

    else:
        print('your selection for stop words set was wrong!\nprogram stopped!')
        sys.exit(-1)
    if stemming:
        stemmer = PersianStemmer()

    colloquial_words = {

        'اگه': 'اگر',
        'میخواین': 'میخواهید',
        'باشین': 'باشید',
        'میتونین': 'میتوانید',
        'میتونید': 'میتوانید',
        'بلوک': 'خیابان'
    }

    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    outlier_count = 0

    for index in range(len(dataframe)):

        current_sentence = str(dataframe['sentence'].iloc[index])

        current_sentence = demoji.replace(current_sentence, ' ')  # Emoji

        current_sentence = re.sub('[\u0653-\u065F\u200c]', '', current_sentence)  # Diacritics and ZWNJ

        current_sentence = re.sub(r'[!@#$%٪^&*(),.?":{}|<>/«»،؛؟…_+-]', ' ', current_sentence)

        current_sentence = re.sub('[\u0660-\u0669\u06F0-\u06F9]', ' ', current_sentence)  # Farsi numbers

        current_sentence = re.sub('[a-zA-Z0-9]+', ' ', current_sentence)  # English letters and numbers

        current_sentence = re.sub(r'[\u2600-\u26FF]', ' ', current_sentence)  # Miscellaneous symbols

        current_sentence = re.sub(' +', ' ', current_sentence)  # Extra spaces

        current_sentence = current_sentence.strip()

        sentence_length = len(current_sentence)

        if remove_outlier:
            if sentence_length < 11 or sentence_length > 600:
                # print(f"this sentence\n{dataframe['sentence'].iloc[index]}\n"
                #       f"at index {index} is outlier with length {sentence_length}")
                outlier_count += 1
                continue

        sentence_tokens = tokenizer.tokenize(current_sentence)

        indexes_to_remove = []

        for j in range(len(sentence_tokens)):

            if sentence_tokens[j] == '[UNK]':
                indexes_to_remove.append(j)
                continue

            if sentence_tokens[j] in colloquial_words:
                sentence_tokens[j] = colloquial_words[sentence_tokens[j]]

            if stemming:
                sentence_tokens[j] = stemmer.stem(sentence_tokens[j])

            if stop_words_type == 'general' and sentence_tokens[j] in general_stop_words:
                indexes_to_remove.append(j)
                continue

            if stop_words_type == 'custom' and sentence_tokens[j] in custom_stop_words:
                indexes_to_remove.append(j)
                continue

        final_sentence = ' '.join(word for index, word in enumerate(sentence_tokens) if index not in indexes_to_remove)
        new_df = new_df.append({'sentence': final_sentence, 'label': dataframe['label'].iloc[index]}, ignore_index=True)
        indexes_to_remove.clear()

    if outlier_count == 0:
        print(f'there is no outlier in data')
    else:
        print(f'there are {outlier_count} outlier in data')
        print(f'current dataframe length before remove {outlier_count} outlier sentence is: {len(dataframe)}')
        print(f'current dataframe length after remove {outlier_count} outlier sentence is: {len(new_df)}')

    new_df = new_df.astype({'label': 'int'})
    return new_df

def remove_emoji(dataframe):
    new_df = pd.DataFrame(columns=['sentence', 'label'])
    for index in range(len(dataframe)):
        current_sentence = demoji.replace(str(dataframe['sentence'].iloc[index]), ' ')
        current_sentence = re.sub(' +', ' ', current_sentence)
        new_df = new_df.append({'sentence': current_sentence, 'label': dataframe['label'].iloc[index]}, ignore_index=True)
    new_df = new_df.astype({'label': 'int'})
    return new_df


def auto_preprocess(dataframe, stop_words_type, stemming):
    new_dataframe = normalize_sentence(dataframe)
    new_dataframe = clean_text(new_dataframe, stop_words_type, stemming)
    print('auto preprocess completed done on dataframe!\nEnjoy!')
    return new_dataframe

