import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim)
        self.keys = nn.Linear(self.head_dim, self.head_dim)
        self.queries = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

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

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(    #用于进一步处理和转换注意力输出
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


class LightweightFeatureExtractor(nn.Module):
    def __init__(self, sequence_length=512, channel=128):
        super(LightweightFeatureExtractor, self).__init__()
        self.sequence_length = sequence_length
        self.channel = channel
        self.features = nn.Sequential(


            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.sequence_length * self.channel)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, self.sequence_length, self.channel)  # 调整形状以符合输出要求
        return x


class MultimodalPromptLearner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dtype = torch.float32
        self.device = args.device
        self.ctx_dim = 512
        self.prefix_n_ctx = args.prefix_length
        self.suffix_n_ctx = args.suffix_length
        self.prompt_size = args.prompt_size
        self.image_size = args.image_size

        self.max_token_length = 77  # Assuming max length of 77 for simplicity; adjust as needed

        self.prompt_generator = PrefixSuffixGenerator(self.ctx_dim, self.ctx_dim)

        self.library_length = 77

        self.prompt_embedding = torch.nn.Embedding(self.library_length, self.ctx_dim)
        # 从图像中提取高维特征
        self.image_previewer = LightweightFeatureExtractor(self.prompt_size, self.ctx_dim)
        self.transformer = TransformerBlock(self.ctx_dim, 4)

        self.avg_1d = nn.AdaptiveAvgPool1d(self.image_size)

    def forward(self, images, actionlist, actiondict, actiontoken):
        class_num = len(actionlist)

        # Initialize the embeddings and token tensors
        text_embedding = torch.zeros(class_num, self.max_token_length, 512).to(self.device)  #[class_num, 77, 512]

        prompt_actiontoken = torch.zeros(class_num, self.max_token_length).to(self.device)  #[class_num, 77]

        # Generate embeddings and tokens for each class
        class_texts = [actiondict[class_name][0] for class_name in actionlist]

        # Convert class_texts to a tensor and move to device
        class_embedding = torch.tensor(class_texts, dtype=torch.float32, device=self.device)

        # Generate prompts with class embeddings
        prompt_embedding = self.prompt_embedding(torch.arange(self.library_length).to(self.device)).repeat([class_num, 1, 1])
        embedding = torch.cat((class_embedding, prompt_embedding), dim=1)

        text_embedding = self.prompt_generator(prompt_embedding, embedding)



        for idx, class_name in enumerate(actionlist):

            embedding = torch.from_numpy(actiondict[class_name][0]).float().to(self.device)
            token = torch.from_numpy(actiontoken[class_name][0]) #token 包含了与当前类别（如 'cat', 'dog', 'bird'）相关的动作token信息

            ind = np.argmax(token, -1)


            end_idx = self.prefix_n_ctx + 1 + ind
            end_idx_with_suffix = self.prefix_n_ctx + ind + self.suffix_n_ctx

            # action embedding
            text_embedding[idx][0] = embedding[0]

            text_embedding[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + 1 + ind] = embedding[1:1 + ind]

            if end_idx_with_suffix < self.max_token_length:
                text_embedding[idx][end_idx_with_suffix] = embedding[ind]

            # action token
            prompt_actiontoken[idx][0] = token[0]
            prompt_actiontoken[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + 1 + ind] = token[1:1 + ind]
            if end_idx_with_suffix < self.max_token_length:
                prompt_actiontoken[idx][end_idx_with_suffix] = token[ind]

            # 手动将超出部分置为0
            if end_idx_with_suffix + 1 < self.max_token_length:
                text_embedding[idx][end_idx_with_suffix + 1:] = 0
                prompt_actiontoken[idx][end_idx_with_suffix + 1:] = 0


        text_embedding.to(self.device)
        prompt_actiontoken.to(self.device)

        batch_size = images.size(0)  # Get the batch size from input images

        #Image
        # 图像和文本特征的初始化
        x: [batch_size, 3, 224,224]
        text_features: [77, 512]

        #batch_size = images.size(0)  # Get the batch size from input images
        #[batch_size, token_length, embedding_dim]
        image_tokens = self.image_previewer(images)  # LightweightFeatureExtractor从输入图像中提取高维特征


        prompt = self.transformer(image_tokens,   #将图像特征与文本嵌入库进行融合
                                  #通过 TransformerBlock 将图像特征 image_tokens 与文本嵌入库进行融合，生成融合后的提示 prompt。
                                  self.prompt_embedding(torch.arange(self.library_length).to(self.device)).repeat(
                                      batch_size, 1, 1),
                                  # self.prompt_embedding 生成的嵌入作为查询、键和值输入到变换器块中，进行跨模态注意力计算。
                                  self.prompt_embedding(torch.arange(self.library_length).to(self.device)).repeat(
                                      batch_size, 1, 1))

        # Reshape the prompt to match the prompt size and image width
        prompt = self.avg_1d(prompt)

        # Concatenate the prompt at the bottom of the image
        image_embedding = torch.cat((images, prompt.unsqueeze(1).repeat(1, 3, 1, 1)), dim=2)  # Shape: [batch_size, 3, image_size + prompt_size, image_size]

        #image_embedding = images

        return image_embedding,text_embedding, prompt_actiontoken




    # def __init__(self, args):
    #     super().__init__()
    #     self.dtype = torch.float32
    #     self.device = args.device
    #     self.ctx_dim = 512
    #     self.prefix_n_ctx = args.prefix_length
    #     self.suffix_n_ctx = args.suffix_length
    #     self.prompt_size = args.prompt_size
    #     self.image_size = args.image_size
    #
    #     self.prompt_generator = PrefixSuffixGenerator(self.ctx_dim, self.ctx_dim)
    #
    #     self.library_length = 128
    #
    #     self.prompt_embedding = torch.nn.Embedding(self.library_length, self.ctx_dim)
    #
    #     self.image_previewer = LightweightFeatureExtractor(self.prompt_size, self.ctx_dim)
    #     self.transformer = TransformerBlock(self.ctx_dim, 4)
    #
    #     self.avg_1d = nn.AdaptiveAvgPool1d(self.image_size)
    #
    #
    # def forward(self, images, actionlist, actiondict, actiontoken):
    #     class_num = len(actionlist)
    #     max_token_length = 77  # Assuming max length of 77 for simplicity; adjust as needed
    #
    #     # Initialize the embeddings and token tensors
    #     text_embedding = torch.zeros(class_num, max_token_length, 512).to(self.device)
    #     prompt_actiontoken = torch.zeros(class_num, max_token_length).to(self.device)
    #
    #     # Generate embeddings and tokens for each class
    #     class_texts = [actiondict[class_name][0] for class_name in actionlist]
    #
    #     # Convert class_texts to a tensor and move to device
    #     class_embedding = torch.tensor(class_texts, dtype=torch.float32, device=self.device)
    #
    #     # Generate prompts with class embeddings
    #     prompt_embedding = self.prompt_embedding(torch.arange(77).to(self.device)).repeat([class_num, 1, 1])
    #     text_embedding = self.prompt_generator(class_embedding, class_embedding)
    #
    #     # prompt_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
    #     # text_embedding = self.prompt_generator(prompt_embedding)
    #     # prompt_actiontoken = torch.zeros(len(actionlist), 77)
    #
    #     for idx, class_name in enumerate(actionlist):
    #         embedding = torch.from_numpy(actiondict[class_name][0]).float().to(self.device)
    #         token = torch.from_numpy(actiontoken[class_name][0])
    #
    #         # 确定每个label对应的分词的长度
    #         ind = np.argmax(token, -1)
    #
    #         # action embedding
    #         # 将startoftext放到开头
    #         text_embedding[idx][0] = embedding[0]
    #         # 将label对应的embedding放入text_embedding
    #         text_embedding[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = embedding[1:ind]
    #         # 将endoftext放到末尾
    #         text_embedding[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = embedding[ind]
    #
    #         # action token
    #         prompt_actiontoken[idx][0] = token[0]
    #         prompt_actiontoken[idx][self.prefix_n_ctx + 1: self.prefix_n_ctx + ind] = token[1:ind]
    #         prompt_actiontoken[idx][self.prefix_n_ctx + ind + self.suffix_n_ctx] = token[ind]
    #
    #     text_embedding.to(self.device)
    #     prompt_actiontoken.to(self.device)
    #
    #     # Image
    #     # pixel_prompt = torch.cat([self.pad_left, self.base, self.pad_right], dim=3)
    #     # pixel_prompt = torch.cat([self.pad_up, pixel_prompt, self.pad_down], dim=2)
    #     # pixel_prompt = torch.cat(images.size(0) * [pixel_prompt])
    #
    #     # x: [batch_size, 3, 224,224]
    #     # text_features: [77, 512]
    #
    #     # batch_size = images.size(0)  # Get the batch size from input images
    #     #
    #     # image_tokens = self.image_previewer(images)
    #     #
    #     # # Apply cross-attention between text features and vector library
    #     # prompt = self.transformer(image_tokens,
    #     #                           self.prompt_embedding(torch.arange(self.library_length).to(self.device)).repeat(
    #     #                               batch_size, 1, 1),
    #     #                           self.prompt_embedding(torch.arange(self.library_length).to(self.device)).repeat(
    #     #                               batch_size, 1, 1))
    #     #
    #     # # Reshape the prompt to match the prompt size and image width
    #     # prompt = self.avg_1d(prompt)
    #     #
    #     # # Concatenate the prompt at the bottom of the image
    #     # image_embedding = torch.cat((images, prompt.unsqueeze(1).repeat(1, 3, 1, 1)),
    #     #                             dim=2)  # Shape: [batch_size, 3, image_size + prompt_size, image_size]
    #
    #     image_embedding = images
    #
    #     return image_embedding, text_embedding, prompt_actiontoken