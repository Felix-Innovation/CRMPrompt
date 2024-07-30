import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np

class TextPromptLearner(nn.Module):
    def __init__(self, args):
        self.device = args.device
        super().__init__()
        self.class_num = args.class_num
        ctx_dim = 512
        self.prefix_n_ctx = args.prefix_length
        self.suffix_n_ctx = args.suffix_length

        self.embedding = torch.nn.Embedding(77, ctx_dim)


    def forward(self, actionlist, actiondict, actiontoken):
        text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        prompt_actiontoken = torch.zeros(len(actionlist), 77)

        for idx, class_name in enumerate(actionlist):
            embedding = torch.from_numpy(actiondict[class_name][0]).float().to(self.device)
            token = torch.from_numpy(actiontoken[class_name][0])

            # 确定每个label对应的分词的长度
            ind = np.argmax(token, -1)

            # action embedding
            # 将startoftext放到开头
            text_embedding[idx][0] = embedding[0]
            # 将label对应的embedding放入text_embedding
            text_embedding[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = embedding[1:ind]
            # 将endoftext放到末尾
            text_embedding[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = embedding[ind]

            # action token
            prompt_actiontoken[idx][0] = token[0]
            prompt_actiontoken[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = token[1:ind]
            prompt_actiontoken[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = token[ind]

        text_embedding.to(self.device)
        prompt_actiontoken.to(self.device)

        # prefix = self.prefix_vectors.unsqueeze(0).expand(self.class_num, -1, -1)
        # suffix = self.suffix_vectors.unsqueeze(0).expand(self.class_num, -1, -1)
        # embeddings = torch.cat([prefix, light_embed, suffix], dim=1)

        return text_embedding, prompt_actiontoken


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        #self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, value_len, key_len, query_len):
        N = query.shape[0]
        value_len, key_len, query_len = value_len, key_len, query_len

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does the matrix multiplication for query*keys for each training ex  ample
        # with every other training example, don't be confused by einsum
        # it's just a way to do bmm with more dimensions
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply softmax and then value
        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        #out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size)
        )

    def forward(self, query, value, key):
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        attention = self.attention(value, key, query, value_len, key_len, query_len)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class PrefixSuffixGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim*2)
        self.fc2 = nn.Linear(output_dim*2, output_dim)
        self.attention = TransformerBlock(512, 8)

    def forward(self, class_embedding, prompt_embedding):
        # input shape : [class_num, 77, 512]
        x = self.attention(class_embedding, prompt_embedding, prompt_embedding)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdaptiveTextPromptLearner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dtype = torch.float32
        self.device = args.device
        self.ctx_dim = 512
        self.prefix_n_ctx = args.prefix_length
        self.suffix_n_ctx = args.suffix_length

        self.prompt_generator = PrefixSuffixGenerator(self.ctx_dim, self.ctx_dim)

        self.prompt_embedding = torch.nn.Embedding(77, self.ctx_dim)


    def forward(self, actionlist, actiondict, actiontoken):
        class_num = len(actionlist)
        max_token_length = 77  # Assuming max length of 77 for simplicity; adjust as needed

        # Initialize the embeddings and token tensors
        text_embedding = torch.zeros(class_num, max_token_length, 512).to(self.device)
        prompt_actiontoken = torch.zeros(class_num, max_token_length).to(self.device)

        # Generate embeddings and tokens for each class
        class_texts = [actiondict[class_name][0] for class_name in actionlist]

        # Convert class_texts to a tensor and move to device
        class_embedding = torch.tensor(class_texts, dtype=torch.float32, device=self.device)

        # Generate prompts with class embeddings
        prompt_embedding = self.prompt_embedding(torch.arange(77).to(self.device)).repeat([class_num, 1, 1])
        text_embedding = self.prompt_generator(class_embedding, prompt_embedding)

        for idx, class_name in enumerate(actionlist):
            embedding = torch.from_numpy(actiondict[class_name][0]).float().to(self.device)
            token = torch.from_numpy(actiontoken[class_name][0])

            # 确定每个label对应的分词的长度
            ind = np.argmax(token, -1)

            # action embedding
            # 将startoftext放到开头
            text_embedding[idx][0] = embedding[0]
            # 将label对应的embedding放入text_embedding
            text_embedding[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = embedding[1:ind]
            # 将endoftext放到末尾
            text_embedding[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = embedding[ind]

            # action token
            prompt_actiontoken[idx][0] = token[0]
            prompt_actiontoken[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = token[1:ind]
            prompt_actiontoken[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = token[ind]

        text_embedding.to(self.device)
        prompt_actiontoken.to(self.device)

        return text_embedding, prompt_actiontoken
