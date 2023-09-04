# -*- coding: utf-8 -*-
"""BERT-Torch
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1LVhb99B-YQJ1bGnaWIX-2bgANy78zAAt
"""

'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, modify by wmathor
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
         https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
'''
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import re

device = torch.device('cuda:0')

text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
# 将每个句子的标点符号去掉，并形成列表
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
# 将每个句子拆分成多个单词，形成列表
word_list = list(set(" ".join(sentences).split())) # ['hello', 'how', 'are', 'you',...]
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
# enumerate函数用于将一个列表中的元素形成索引序列，例如：l=['a','b','c']，enumerate(l)=(0,'a'),(1,'b'),(2,'c')
# 将每个词都设置一个索引，和PAD，CLS，SEP，MASK一起形成一个字典，形式为“单词：索引”
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
# 与word2idx相反，其形式为“索引：单词”
idx2word = {i: w for i, w in enumerate(word2idx)}
# 词表大小
vocab_size = len(word2idx)


# 获取每个句子中每个词语在词表中的的索引，形成列表
token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

# BERT Parameters
maxlen = 30 # 表示每个batch中的所有句子都由30个token组成，不够的补PAD
batch_size = 6
max_pred = 5 # max tokens of prediction，最多需要预测多少个词语
n_layers = 6 # 表示encoder layer的数量
n_heads = 12 # 是指Multi-Head-Attention中self-Attention的个数
d_model = 768 # 表示Token Embedding，Segment Embedding、Position Embedding的维度
d_ff = 768*4 # 4*d_model, FeedForward dimension ，表示Encoder layer中全连接层的维度
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2 # 表示Decoder input由几句话组成



# 数据预处理部分，需要mask一句话中15%的token，还需要随机拼接两句话
# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            len(sentences))  # sample random index in sentences
        # 取出这两个句子的单词索引
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        # 将随机选取的两个句子的单词索引拼接在一起，而且加入CLS和SEP标记，此时input_ids中每个元素表示单词在词表中的索引
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        # 组成Segment Embedding
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence，确定要mask的单词数量
        # 此时cand_maked_pos表示在input_ids中有哪几个位置可以被mask，这个位置是指在此列表中的位置，而不是在词汇表中的索引。
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx[
                              '[SEP]']]  # candidate masked position，选择出可以mask的位置的索引，因为像CLS和SEP这些位置不可以被mask
        # shuffle将序列中的元素随机排序，实现随机选取单词来mask
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            # masked_pos 表示要mask的单词在input_ids中的位置，而不是在词表中的索引
            masked_pos.append(pos)
            # masked_tokens表示要mask的单词在此表中的索引，因为input_ids中存的就是选取的两个句子的单词索引
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask，进行mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary，替换为词表中一个随机的单词
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace

        # Zero Paddings，使得一个batch中的句子都是相同长度
        n_pad = maxlen - len(input_ids)  # 需要补的0的个数
        input_ids.extend([0] * n_pad)  # extend函数会在列表末尾一次性添加另一个序列的多个值
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens，补齐mask的序列，保证一个batch中所有句子mask的数量是一样的
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 正样本，即两个句子是相连的
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        # 负样本，而且要保证正负样本的比例是1:1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch


# Proprecessing Finished

batch = make_data()
# 将batch中的数据分开，input_ids, segment_ids, masked_tokens, masked_pos, isNext分别存到一个单独的集合中
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.Tensor(input_ids).to(device), torch.Tensor(segment_ids).to(device), torch.Tensor(masked_tokens).to(device), \
    torch.Tensor(masked_pos).to(device), torch.Tensor(isNext).to(device)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)


# 将之前补0的位置mask掉，让其不参与运算
def get_attn_pad_mask(seq_q):
    # batch_size就是上述定义的6，seq_len即句子长度30
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    # seq_q.data.eq(0)可以找出句子中哪些位置是PAD标记法，返回的数据与seq_q的维度相同。然后用unsqueeze来扩充维度
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    # 扩充维度后再将其变形为[batch_size, seq_len, seq_len]的维度
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


# 激活函数
def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # Embedding模块主要有两个参数，第一个是单词本中的单词个数，第二个是输出矩阵的大小
        self.tok_embed = nn.Embedding(vocab_size, d_model).to(device)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model).to(device)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model).to(device)  # segment(token type) embedding
        # LayerNorm是一个归一化层
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x, seg):
        # size函数输出矩阵的某个维度
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        # pos = pos.unsqueeze(0)首先将pos扩充为二维矩阵，然后expand_as将pos扩充为与x维度相同的矩阵
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        # 计算最终输入的Embedding，此时的Embedding是的维度为[batch_size,max_len,d_model]
        embedding = self.tok_embed(x.long()) + self.pos_embed(pos.long()) + self.seg_embed(seg.long())
        return self.norm(embedding)


# 计算self-Attention的输出
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        # 在归一化的时候，scores中的0也会有一个值，会影响最终的结果，所以将之前补0的位置替换为一个非常小的负数，不让它影响softmax的结果
        scores.masked_fill_(attn_mask,
                            -1e9)  # Fills elements of self tensor with value where mask is one.将scores矩阵中attn_mask上值为true所对应的位置填充为-1e9，也就是那些补0的位置
        # 这里dim设置为-1，表示对某一维度进行归一化
        # self-Attention的输出矩阵，是要对一个单词对其他所有单词的attention系数进行归一化，所以是在同一维度上的。不是同一位置(0)，也不是同一列(1)，也不是同一行(2)，所以dim设置为-1
        attn = nn.Softmax(dim=-1)(scores)
        # context的维度为：[batch_size,n_head,seq_len,d_k]
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 线性变换矩阵
        self.W_Q = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_heads).to(device)
        self.linear = nn.Linear(n_heads * d_v, d_model).to(device)
        self.normal = nn.LayerNorm(d_model).to(device)

    def forward(self, Q, K, V, attn_mask):
        # seq_len表示句子的长度，即q，k，v的行数是句子中的单词书，列数是我们自己设置d_model
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 多个self-Attention就要生成多维的q,k,v。transpose用于交换矩阵的两个维度
        # view函数中的参数-1表示该维度的维数由其他维度来估算得到，也就是说这个维度的维数会由程序自动计算
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]
        # unsqueeze对数据维度进行扩充
        # repeat指在某个维度进行重复，repeat(1,n_heads,1,1)表示在第二个维度上重复n_heads次。使pad_mask保持与q，k，v相同的维度
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        # 计算Multi-Head-Attention的输出，但此时还不是最终输出，还没有将多个self-Attention的输出拼接起来
        model = ScaledDotProductAttention()
        # model=model.to(device)
        context = model(q_s, k_s, v_s, attn_mask)
        # 交换第一维度和第二维度的维数，transpose函数在交换时并不会重新开辟一块内存来存储转换后的数据，而是保持原有数据存放位置不变修改一些行列的对应关系
        # 也就是说，经过transpose后，两个矩阵实际共享同一块内存，修改一个矩阵，另一个矩阵的值也会随之变化。
        # contiguous函数会将转换后的矩阵，按照其维度来从头开辟一块内存，并原模原样存放该矩阵，不再共享内存
        # n_heads * d_v就表示将多个self-Attention的输出矩阵拼接起来
        context = context.transpose(1, 2).contiguous().to(device).view(batch_size, -1,
                                                                       n_heads * d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        # 进行线性变换
        output = self.linear(context)
        # print(output)
        # 经归一化后生成最终输出，且输出矩阵与输入矩阵的维度是一样的
        # output+ residual表示残差连接
        return self.normal(output + residual)  # output: [batch_size, seq_len, d_model]


# feedforward是一个两层的全连接层，第一层使用gelu激活函数，第二层不使用激活函数
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff).to(device)
        self.fc2 = nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    # 每层Encoder都要先经过Multi-Head Attention，再经过feed-forword。而且两者上方都有一个Norm层，用来对每层的激活值进行归一化
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        # 建立6层的Encoder
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)
        # Sequential是一个有序的容器，神经网络的各种模块在这里面被顺序添加执行
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        ).to(device)
        self.classifier = nn.Linear(d_model, 2).to(device)
        self.linear = nn.Linear(d_model, d_model).to(device)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False).to(device)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        # 得到transformer的输入Embedding
        output = self.embedding(input_ids, segment_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            # 前一层Encoder的输出作为后一层Encoder的输入
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        # output[: ,0]取出每个句子中CLS的所有attention系数
        h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
        # 用第一个位置CLS的output丢进Linear classifier来预测一个class
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        # masked_pos是每句话中要预测的单词的位置，将去扩充到与output相同的维度
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        # 按照masked_pos的值，抽取出output中对应索引的数据
        h_masked = torch.gather(output, 1,
                                masked_pos.to(torch.int64))  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf


model = BERT()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss().cuda()
# 定义优化器，将bert的模型参数传入进行优化
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
for epoch in range(500):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1).long())  # for masked LM
        # mean表示求平均值
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext.long())  # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 5 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        # 清空梯度
        optimizer.zero_grad()
        # 计算反向传播
        loss.backward()
        optimizer.step()

# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.Tensor([input_ids]).to(device),
                               torch.Tensor([segment_ids]).to(device), torch.Tensor([masked_pos]).to(device))
logits_lm = logits_lm.data.max(2)[1][0].data.cpu().numpy()
print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.cpu().numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ', True if logits_clsf else False)