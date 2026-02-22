import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=1536, dropout_rate=0.3):
        super(TwoTowerModel, self).__init__()
        
        tower_input_dim = input_dim // 2
        
        # Query tower (user)
        self.user_tower = self.build_tower(tower_input_dim, dropout_rate)
        # Candidate tower (news)
        self.news_tower = self.build_tower(tower_input_dim, dropout_rate)
        
        classifier_input_dim = 384
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.apply(self.init_weights)

    # Helper function to build MLP towers
    def build_tower(self, input_dim, drop_rate):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.5)
        )
    
    # Deeper Tower Model
    # def build_tower(self, input_dim, drop_rate):
    #     return nn.Sequential(
    #         nn.Linear(input_dim, 512),
    #         nn.BatchNorm1d(512),
    #         nn.ReLU(),
    #         nn.Dropout(drop_rate),

    #         nn.Linear(512, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Dropout(drop_rate * 0.75),
            
    #         nn.Linear(256, 128),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(),
    #         nn.Dropout(drop_rate * 0.5)
    #     )
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, v_user, v_news):
        emb_user = self.user_tower(v_user)
        emb_news = self.news_tower(v_news)
        merged = torch.cat([emb_user, emb_news, emb_user * emb_news], dim=1)
        return self.classifier(merged)
    
class DNNModel(nn.Module):
    def __init__(self, input_dim=1536, dropout_rate=0.3):
        super(DNNModel, self).__init__()
        
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, v_user, v_news):
        combined_input = torch.cat([v_user, v_news], dim=1)
        output = self.network(combined_input)
        return output

class WideAndDeepModel(nn.Module):
    def __init__(self, input_dim=1536, dropout_rate=0.3):
        super(WideAndDeepModel, self).__init__()
        
        self.tower_input_dim = input_dim // 2 
        
        # Deep Part
        self.user_tower = self.build_tower(self.tower_input_dim, dropout_rate)
        self.news_tower = self.build_tower(self.tower_input_dim, dropout_rate)
        
        # Wide Part
        self.wide_linear = nn.Linear(1, 1) 
        
        self.apply(self.init_weights)

    def build_tower(self, input_dim, drop_rate):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, v_user, v_news):
        u_deep = self.user_tower(v_user)
        if v_news.dim() == 3: 
            B, K, D = v_news.shape
            n_deep = self.news_tower(v_news.view(B*K, D)).view(B, K, -1)
            deep_score = (u_deep.unsqueeze(1) * n_deep).sum(dim=2, keepdim=True) 

            u_raw = v_user.unsqueeze(1)
            raw_dot = (u_raw * v_news).sum(dim=2, keepdim=True)
            wide_score = self.wide_linear(raw_dot)

            logits = (deep_score + wide_score).squeeze(-1)
            return torch.sigmoid(logits) 
            
        else: 
            n_deep = self.news_tower(v_news)
            deep_score = (u_deep * n_deep).sum(dim=1, keepdim=True)
            raw_dot = (v_user * v_news).sum(dim=1, keepdim=True)
            wide_score = self.wide_linear(raw_dot)
            logits = deep_score + wide_score
            return torch.sigmoid(logits)
        
class LSTMTwoTowerModel(nn.Module):
    def __init__(self, input_dim=768, lstm_hidden=256, dropout_rate=0.3):
        super().__init__()
        
        self.user_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False
        )
        
        self.user_projection = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        self.news_tower = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, user_seq, news_vec):
        lstm_out, (hidden, cell) = self.user_lstm(user_seq)
        user_repr = hidden[-1]
        
        user_emb = self.user_projection(user_repr)
        news_emb = self.news_tower(news_vec)
        interaction = user_emb * news_emb

        merged = torch.cat([user_emb, news_emb, interaction], dim=1)
        return self.classifier(merged)