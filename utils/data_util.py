import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import pickle
import json
import os


max_len = 20
word_threshold = 2
counter = None


def load_features(feature_path):
    features = np.load(feature_path)
    features = np.repeat(features, 5, axis=0)
    features = np.array([feat.astype(float) for feat in features])
    print("Features Loaded", feature_path)
    return features


def get_data(required_files):
    ret = []
    for fil in required_files:
        ret.append(np.load("Dataset/" + fil + ".npy"))

    training_data = ret[2]
    features, captions = training_data[:, 0], training_data[:, 1]
    struct_features = np.zeros((len(features), 1536))
    for idx, row in enumerate(features):
        struct_features[idx] = row

    ret[0] = ret[0].tolist()
    ret[1] = ret[1].tolist()
    ret[2] = struct_features
    ret.append(captions)
    return ret


def preprocess_flickr_captions(filenames, captions):
    global max_len
    print("Preprocessing Captions...")
    cap_token_df = pd.DataFrame()
    cap_token_df['FileNames'] = filenames
    cap_token_df['caption'] = captions
    cap_token_df['caption'] = cap_token_df['caption'].apply(word_tokenize).apply(
        lambda x: x[:max_len]).apply(" ".join).str.lower()
    return cap_token_df

def split_dataset(df, features, ratio=0.8):
    split_idx = int(df.shape[0] * ratio)
    print("Data Statistics:")
    print("# Records Total Data: ", df.shape[0])
    print("# Records Training Data: ", split_idx)
    print("# Records Validation Data: ", df.shape[0] - split_idx)
    print("Ration of Training: Validation = ", ratio * 100, ":", 100 - (ratio * 100))
    val_features = features[split_idx:]
    val_captions = np.array(df.caption)[split_idx:]
    np.save("Dataset/valid_data", zip(val_features, val_captions))
    return df[:split_idx], features[:split_idx]


def generate_vocab(capt_token_df):
    global max_len, word_threshold, counter
    print("Generating Vocabulary")

    vocab = dict([w for w in counter.items() if w[1] >= word_threshold])
    vocab["<UNK>"] = len(counter) - len(vocab)
    vocab["<PAD>"] = capt_token_df.caption.str.count("<PAD>").sum()
    vocab["<S>"] = capt_token_df.caption.str.count("<S>").sum()
    vocab["</S>"] = capt_token_df.caption.str.count("</S>").sum()

    wtoidx = {}
    wtoidx["<S>"] = 1
    wtoidx["</S>"] = 2
    wtoidx["<PAD>"] = 0
    wtoidx["<UNK>"] = 3
    print("Generating Word to Index and Index to Word")
    i = 4
    for word in vocab.keys():
        if word not in ["<S>", "</S>", "<PAD>", "<UNK>"]:
            wtoidx[word] = i
            i += 1
    print("Size of Vocabulary", len(vocab))
    return vocab, wtoidx


def pad_captions(capt_token_df):
    global max_len
    print("Padding Caption <PAD> to Max Length", max_len, "+ 2 for <S> and </S>")
    capt_dfPadded = capt_token_df.copy()
    capt_dfPadded['caption'] = "<S> " + capt_dfPadded['caption'] + " </S>"
    max_len = max_len + 2
    for i, row in capt_dfPadded.iterrows():
        capt = row['caption']
        capt_len = len(capt.split())
        if(capt_len < max_len):
            pad_len = max_len - capt_len
            pad_buf = "<PAD> " * pad_len
            pad_buf = pad_buf.strip()
            capt_dfPadded.set_value(i, 'caption', capt + " " + pad_buf)
    return capt_dfPadded


def generate_captions(config,
    capt_path='Dataset/Flickr8k_text/Flickr8k.token.txt',
    feat_path='Dataset/features.npy'):

    required_files = ["vocab", "wordmap", "train_data"]
    generate = False
    for f in required_files:
        if not os.path.isfile('Dataset/' + f + ".npy"):
            generate = True
            print("Required Files not present. Regenerating Data.")
            break
    if not generate:
        print("Dataset Present; Skipping Generation.")
        return get_data(required_files)

    global max_len, word_threshold, counter
    max_len = config.max_len
    word_threshold = config.word_threshold

    print("Loading Caption Data", capt_path)
    with open(capt_path, 'r') as f:
        data = f.readlines()
    filenames = [capts.split('\t')[0].split('#')[0] for capts in data]
    captions = [capts.replace('\n', '').split('\t')[1] for capts in data]
    capt_token_df = preprocess_flickr_captions(filenames, captions)

    features = load_features(feat_path)
    print(features.shape, capt_token_df.shape)
    index = np.random.permutation(features.shape[0])
    capt_token_df = capt_token_df.iloc[index]
    features = features[index]
    capt_token_df, features = split_dataset(capt_token_df, features)
    counter = Counter()
    for i, row in capt_token_df.iterrows():
        counter.update(row["caption"].lower().split())
    capt_token_df = pad_captions(capt_token_df)
    vocab, wtoidx = generate_vocab(capt_token_df)
    captions = np.array(capt_token_df.caption)

    np.save("Dataset/train_data", list(zip(features, captions)))
    np.save("Dataset/wordmap", wtoidx)
    np.save("Dataset/vocab", vocab)
    print(len(wtoidx))

    print("Preprocessing Complete")
    return get_data(required_files)
