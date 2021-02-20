import os

import pandas as pd


def preprocess_text(x):
    for punct in '"!R&}-/<>#$%()*+:;=?@[\\]^_`|~1234567890':
        x = x.replace(punct, ' ')

    x = ' '.join(x.split())
    x = x.lower()

    return x


def create_utterances(filename, split):
    sentences, act_labels, emotion_labels, speakers, conv_id, utt_id = [], [], [], [], [], []

    # lengths = []
    with open(filename, 'r') as f:
        for c_id, line in enumerate(f):
            s = eval(line)
            for u_id, item in enumerate(s['dialogue']):
                sentences.append(item['text'])
                act_labels.append(item['act'])
                emotion_labels.append(item['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
                speakers.append(str(u_id % 2))

                # u_id += 1

    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))

    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id

    return data


if __name__ == '__main__':
    path = 'dailydialog'
    print(os.listdir(path))
    train = create_utterances(path + '/train.json', 'train')
    valid = create_utterances(path + '/valid.json', 'valid')
    test = create_utterances(path + '/test.json', 'test')
    print(train.shape, train.columns.values)
    print(valid.shape, valid.columns.values)
    print(test.shape, test.columns.values)

    print(set(train['emotion_label']))
    print(set(valid['emotion_label']))
    print(set(test['emotion_label']))
    pass
