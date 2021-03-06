import torch
from torchtext import data
import torchtext
import codecs

def load_pairs():


    TEXT1 = data.Field(fix_length=500)
    TEXT2 = data.Field(fix_length=500)
    LABEL = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float64)
    ID = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float64)
    ONEHOT = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float32)

    # TEXT1是对象，返回到数据结构中的应该是个对象类型，而不是单纯的文本类型
    field = {'label': ('label', LABEL), 'text1': ('text1', TEXT1), 'text2': ('text2', TEXT2),
             'onehot1':('onehot1', ONEHOT), 'onehot2':('onehot2', ONEHOT)}
    field1 = {'id': ('id', ID), 'text': ('text', TEXT1), 'label': ('label', LABEL),
              'onehot':('onehot', ONEHOT)}
    # train_pairs 的数据格式为field字典，所以train_pairs有text1键，  然后他的属性值以为Text1
    train_pairs, valid_pairs = data.TabularDataset.splits(  # 切分语料库
        path='./data/',
        train='train_pairs.json',
        validation='val_pairs.json',
        format='json',
        fields=field
    )
    train_data, test_data = data.TabularDataset.splits(
        path='./data/',
        train='compare_data_5.json',
        test='test_data.json',
        format='json',
        fields=field1
    )

    vectors = torchtext.vocab.Vectors(name='./data/fasttext.vec')
    TEXT1.build_vocab(train_pairs, vectors=vectors)
    # build_vocab构建语料库的vocabulary同时加载word-embedding
    # 从预训练的vectors中，将当前corpus词汇表的词向量抽取出来，构成当前corpus的Vocab（词汇表）
    # 自动构建embedding矩阵
    TEXT2.build_vocab(train_pairs, vectors=vectors)


    print('Length of TEXT1 Vocabulary:' + str(len(TEXT1.vocab)))
    print('Length of TEXT2 Vocabulary:' + str(len(TEXT2.vocab)))
    print('Dim of TEXT1,TEXT2:', TEXT1.vocab.vectors.size()[1], TEXT2.vocab.vectors.size()[1])

    train_pairs_iter, valid_pairs_iter = data.BucketIterator.splits(
        (train_pairs, valid_pairs),
        sort=False,
        batch_size=100,
        repeat=False,
        shuffle=True,
        device=torch.device('cuda:0')
    )
    train_data_iter = data.BucketIterator(
        train_data,
        sort=False,
        batch_size=5,
        repeat=False,
        shuffle=False,
        device=torch.device('cuda:0')
    )
    test_data_iter = data.BucketIterator(
        test_data,
        sort=False,
        batch_size=100,
        repeat=False,
        shuffle=True,
        device=torch.device('cuda:0')
    )

    return train_pairs_iter, valid_pairs_iter, train_data_iter, test_data_iter

def load_label_list():

    label_list = []
    f = codecs.open('/data/label.txt','r')
    for line in f:
        line = line.strip()
        label_list.append(line)


    return label_list



















