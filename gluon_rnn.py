import random
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from math import exp
import os
import time

import numpy as np
import collections
import random

poetry_file='p.txt'

BEGIN_CHAR = '^' 
END_CHAR = '$' 
UNKNOWN_CHAR = '*' 
MAX_LENGTH = 100 
MIN_LENGTH = 10 
max_words = 6000

class Data:    
    def __init__(self):

        self.batch_size = 64
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    def load(self):

        def handle(line):
            if len(line) > MAX_LENGTH:

                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]

            return BEGIN_CHAR + line + END_CHAR


        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        open(self.poetry_file)]

        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]

        # 所有字
        words = []

        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)
        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)
        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]
      
    def create_batches(self):
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            length = max(map(len, batches))
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r

            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]

            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


p_data=Data()
char_to_idx=p_data.char2id
idx_to_char=p_data.id2char
vocab_size=p_data.words_size
# with open('p.txt') as f:
#     corpus_chars = f.read()

# corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').replace('__', ' ').replace('，', ' ').replace('。', ' ').replace(':', ' ').replace('？', ' ').replace('、', ' ')
# corpus_chars=''.join(corpus_chars.split())

# idx_to_char = list(set(corpus_chars))
# char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
# vocab_size = len(char_to_idx)
# #print(vocab_size)
# corpus_indices = [char_to_idx[char] for char in corpus_chars]
# #print(len(corpus_indices))

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为label的索引是相应data的索引加一
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个数据
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    # 减一是因为label的索引是相应data的索引加一
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


def predict(net,prefix, num_chars,ctx):
    # 预测以 prefix 开始的接下来的 num_chars 个字符
    prefix = prefix.lower()
    output = list(map(p_data.char2id, prefix[0]))

    for i in range(num_chars + len(prefix)-1):

        X = nd.array([output[-1]], ctx=ctx).reshape((1,1))
        Y = net(X)

        if i < len(prefix)-1:
            next_input = list(map(p_data.char2id, prefix[i+1]))[0]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())

        if list(map(p_data.id2char, [next_input]))[0] in ['，','。',BEGIN_CHAR,END_CHAR]:
            # output.append(next_input)
            next_input=random.randint(0,3000)

        output.append(next_input)

    return ''.join(list(map(p_data.id2char, output)))


ctx=mx.gpu()
num_inputs, emb_dim, num_hiddens, num_outputs = vocab_size,128,128, vocab_size #128 64
epochs = 500
# seq_len = 10
batch_size = p_data.batch_size
# dropout=0.5
lr=0.01
clipping_norm=5

seq1 = '明'
seq2 = '月'
seq3 = '别'
seq4='枝'
seq5='惊'
seq6='鹊'

loss = gluon.loss.SoftmaxCrossEntropyLoss()

model = mx.gluon.nn.Sequential()
with model.name_scope():
    model.add(mx.gluon.nn.Embedding(num_inputs, emb_dim))
    
    # model.add(mx.gluon.nn.Dropout(0.8))
    model.add(mx.gluon.rnn.LSTM(num_hiddens,bidirectional=True))
    model.add(mx.gluon.rnn.LSTM(num_hiddens,bidirectional=True))
    # model.add(mx.gluon.nn.Dropout(0.8))

    model.add(mx.gluon.nn.Dense(num_inputs, flatten=False))
model.initialize(mx.init.Xavier(),ctx=ctx)

model.collect_params().load('model/loss_60.5339_lstm.params',ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'sgd',{'learning_rate': lr, 'momentum': 0.9, 'wd': 0})
# trainer = gluon.Trainer(model.collect_params(), 'Adam')

def train(is_random_iter):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive

    loss_min=100
    for epoch in range(epochs):
        if epoch!=0 and epoch%50==0:
            trainer.set_learning_rate(trainer.learning_rate*0.5)
            print('lr:'+str(trainer.learning_rate))
        total_L = 0.0
        num_examples=0
        start_time = time.time()

        for data,label in zip(p_data.x_batches,p_data.y_batches):
            data=nd.array(data,ctx=ctx)
            _,seq_len=data.shape
            # data=data.T

            label=nd.array(label,ctx=ctx)
            label=label.as_in_context(ctx)
     
            with autograd.record():

                outputs = model(data)
                outputs=outputs.reshape((-1,outputs.shape[-1]))

                label = label.reshape((-1,))
                L = loss(outputs, label)

            # print((outputs.argmax(axis=1)==label).sum()/label.shape[0])

            L.backward()
            grads = [x.grad(ctx) for x in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
            # 因此我们将clipping_norm乘以num_steps和batch_size。
            gluon.utils.clip_global_norm(grads, clipping_norm*seq_len*batch_size)
            trainer.step(seq_len*batch_size)
            total_L += mx.nd.sum(L).asscalar()
            num_examples += L.size

        aver_L_exp=exp(total_L/num_examples)
        if loss_min>aver_L_exp:
            if os.path.exists('model/loss_%.4f_lstm.params'%(loss_min)):
                os.remove('model/loss_%.4f_lstm.params'%(loss_min))
            loss_min=aver_L_exp
            model.collect_params().save('model/loss_%.4f_lstm.params'%(loss_min))
        end_time=time.time()
        print("epoch: %d, loss: %.4f, time: %.2f sec"%(epoch+1,aver_L_exp,end_time-start_time))

        print(' - ', predict(model,seq1, 4, ctx))
        print(' - ', predict(model,seq2, 4, ctx))
        print(' - ', predict(model,seq3, 6, ctx))
        print(' - ', predict(model,seq4, 6, ctx))
        print(' - ', predict(model,seq5, 6, ctx))
        print(' - ', predict(model,seq6, 6, ctx), '\n')

train(is_random_iter=True)
# for i in  range(10):
# print(' - ', predict(model,'我', 7, ctx),'\n')
# print(' - ', predict(model,'真', 7, ctx),'\n')
# print(' - ', predict(model,'是', 7, ctx),'\n')
# print(' - ', predict(model,'帅', 7, ctx), '\n')