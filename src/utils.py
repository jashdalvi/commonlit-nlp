import numpy as np
import torch
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_mcrmse(preds, labels):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    if len(col_rmse) > 1:
        mcrmse = np.mean(col_rmse)

        return {
            "content_rmse": col_rmse[0],
            "wording_rmse": col_rmse[1],
            "mcrmse": mcrmse,
        }
    else:
        mcrmse = col_rmse[0]

        return {
            "mcrmse": mcrmse,
        }
    
class MeanPooling(nn.Module):
    """Mean pooling representation"""
    def __init__(self, hidden_size = 768, num_classes = 2, calc_output = True, config = None):
        super(MeanPooling, self).__init__()
        self.calc_output = calc_output
        self.config = config
        if self.calc_output:
            self.output = nn.Linear(hidden_size, num_classes)
            self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        if self.calc_output:
            mean_embeddings = self.output(mean_embeddings)
        return mean_embeddings
    

class LSTMPooling(nn.Module):
    """LSTM Pooling representation"""
    def __init__(self, hidden_size = 768, num_classes = 2, config = None):
        super(LSTMPooling, self).__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size//2, batch_first=True, bidirectional=True)
        self.mean_pooling = MeanPooling(hidden_size=hidden_size, num_classes=num_classes, config = self.config)

    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        last_hidden_state, (_, _) = self.lstm(last_hidden_state)
        mean_embeddings = self.mean_pooling(last_hidden_state, attention_mask, all_hidden_states)
        return mean_embeddings

class MaxPooling(nn.Module):
    """Max pooling representation"""
    def __init__(self, hidden_size = 768, num_classes = 2, calc_output = True, config = None):
        super(MaxPooling, self).__init__()
        self.calc_output = calc_output
        self.config = config
        if self.calc_output:
            self.output = nn.Linear(hidden_size, num_classes)
            self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings, _ = torch.max(last_hidden_state, 1)
        if self.calc_output:
            max_embeddings = self.output(max_embeddings)
        return max_embeddings
    
class MeanMaxPooling(nn.Module):
    """Mean Max pooling representation"""
    def __init__(self, hidden_size = 768, num_classes = 2, config = None):
        super(MeanMaxPooling, self).__init__()
        self.mean_pooling = MeanPooling(hidden_size=hidden_size, num_classes=num_classes, calc_output=False, config = config)
        self.max_pooling = MaxPooling(hidden_size=hidden_size, num_classes=num_classes, calc_output=False, config = config)
        self.output = nn.Linear(hidden_size*2, num_classes)
        self.config = config
        self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        mean_embeddings = self.mean_pooling(last_hidden_state, attention_mask, all_hidden_states)
        max_embeddings = self.max_pooling(last_hidden_state, attention_mask, all_hidden_states)
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)
        mean_max_embeddings = self.output(mean_max_embeddings)
        return mean_max_embeddings
    
class CLSPooling(nn.Module):
    def __init__(self, hidden_size = 768, num_classes = 2, config = None):
        super(CLSPooling, self).__init__()
        self.output = nn.Linear(hidden_size, num_classes)
        self.config = config
        self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        cls_embeddings = last_hidden_state[:, 0, :]
        cls_embeddings = self.output(cls_embeddings)
        return cls_embeddings


class ConcatPooling(nn.Module):
    def __init__(self, hidden_size = 768, num_classes = 2, num_layers = 4, pooling = "cls", config = None):
        super(ConcatPooling, self).__init__()
        self.num_layers = num_layers
        self.config = config
        if pooling == "cls":
            self.pooling = CLSPooling(hidden_size=hidden_size * num_layers, num_classes=num_classes, config = self.config)
        elif pooling == "mean":
            self.pooling = MeanPooling(hidden_size=hidden_size * num_layers, num_classes=num_classes, config = self.config)
        elif pooling == "max":
            self.pooling = MaxPooling(hidden_size=hidden_size * num_layers, num_classes=num_classes, config = self.config)
        elif pooling == "mean_max":
            self.pooling = MeanMaxPooling(hidden_size=hidden_size * num_layers, num_classes=num_classes, config = self.config)

    def forward(self, last_hidden_state, attention_mask, all_hidden_states):
        concat_embeddings = torch.cat([all_hidden_states[-i] for i in range(1, self.num_layers+1)], -1)
        concat_embeddings = self.pooling(concat_embeddings, attention_mask, all_hidden_states)
        return concat_embeddings
    
#Getting the prompt tuning soft embeddings
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)