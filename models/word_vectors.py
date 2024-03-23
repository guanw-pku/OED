"""
Adapted from PyTorch's text library.
"""

import array
import os
import zipfile

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys

def obj_edge_vectors(names, wv_type='glove.6B', wv_dir=None, wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        count = 0
        for word in token.split('/'):
            word_idx = wv_dict.get(word, None)
            if word_idx is not None:
                vectors[i] += wv_arr[word_idx]
                count += 1
        if count > 0:
            vectors[i] /= count
        else:
            print("failed to find any word in %s" % token)

    return vectors


def predicate_edge_vectors(names, wv_type='glove.6B', wv_dir=None, wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)
    for i, tokens in enumerate(names):
        count = 0
        for token in tokens:
            token_idx = wv_dict.get(token, None)
            if token_idx is not None:
                vectors[i] += wv_arr[token_idx]
                count += 1
        if count > 0:
            vectors[i] /= count
        else:
            print("failed to find any word in %s" % token)

    return vectors




URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt)
        except Exception as e:
            print("""
                Error loading the model from {}

                This could be because this code was previously run with one
                PyTorch version to generate cached data and is now being
                run with another version.
                You can try to delete the cached files on disk (this file
                  and others) and re-running the code

                Error message:
                ---------
                {}
                """.format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: ĺeftright].
        bsize: int, optional
        Size of each block (in tqdm units) [default: ĺeftright].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
