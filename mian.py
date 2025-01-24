import torch
import torch.nn as nn
import os
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

# 定义保存数据集的函数
def save_dataset_to_file(dataset, file_name):
    df = pd.DataFrame(dataset)
    df.to_csv(file_name, index=False, sep='\t')

# 保存最好的模型
def save_best_model(model, accuracy, best_accuracy, epoch, model_save_path):
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 保存模型
        torch.save(model.state_dict(), model_save_path)
        print(f"Saving best model at epoch {epoch+1} with accuracy {accuracy:.4f}")
    return best_accuracy

# 评估指标
def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1_binary = f1_score(labels, preds, zero_division=0)
    
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1_binary, f1_macro, f1_micro, f1_weighted

# 自定义数据集类
class ClaimDataset(Dataset):
    def __init__(self, claims, analyses, reasons_true, reasons_false, labels, tokenizer, max_length):
        self.claims = claims
        self.analyses = analyses
        self.reasons_true = reasons_true
        self.reasons_false = reasons_false
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        claim_input = f"[CLS] claim: {self.claims[idx]} [SEP]"
        analyse_input = f"[CLS] {self.analyses[idx][0]} [SEP] {self.analyses[idx][1]} [SEP] {self.analyses[idx][2]} [SEP] {self.analyses[idx][3]} [SEP]"
        true_input = f"[CLS] Supportive Reason: {self.reasons_true[idx]} [SEP]"
        false_input = f"[CLS] Refuted Reason: {self.reasons_false[idx]} [SEP]"

        claim_encoding = self.tokenizer(claim_input, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        analyse_encoding = self.tokenizer(analyse_input, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        true_encoding = self.tokenizer(true_input, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        false_encoding = self.tokenizer(false_input, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
    
        return {
            'claim_input_ids': claim_encoding['input_ids'].squeeze(),
            'claim_attention_mask': claim_encoding['attention_mask'].squeeze(),
            'analyse_input_ids': analyse_encoding['input_ids'].squeeze(),
            'analyse_attention_mask': analyse_encoding['attention_mask'].squeeze(),
            'true_input_ids': true_encoding['input_ids'].squeeze(),
            'true_attention_mask': true_encoding['attention_mask'].squeeze(),
            'false_input_ids': false_encoding['input_ids'].squeeze(),
            'false_attention_mask': false_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# 模型定义
class SemanticAttentionClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, num_classes=2):
        super(SemanticAttentionClassifier, self).__init__()
        model_path = 'bert-base-uncased/'
        self.bert_c = BertModel.from_pretrained(model_path)
        self.bert_t = BertModel.from_pretrained(model_path)
        self.bert_r = BertModel.from_pretrained(model_path)
        self.embedding = self.bert_c.embeddings
        # self.attention_w = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.5)
        # self.analyse_att =nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.5)
        # self.true_att =nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.5)
        # self.false_att =nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.5)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True, dropout=0.5)
        # self.transformer1 = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, dropout=0.5, batch_first=True)
        # self.transformer2 = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, dropout=0.5, batch_first=True)
        # self.transformer_t = nn.Sequential(
        #     nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True) 
        # )
        # self.transformer_f = nn.Sequential(
        #     nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True) 
        # )
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True),
            nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True) 
        )
        self.transformer_c = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True),
            nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=4*hidden_size, batch_first=True) 
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.fc1 = nn.Linear(hidden_size*6, hidden_size)
        self.fc2 = nn.Linear(768, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, claim_input_ids, claim_attention_mask, analyse_input_ids, analyse_attention_mask, true_input_ids, true_attention_mask, false_input_ids, false_attention_mask, labels):

        claim_embedding = self.bert_c(input_ids=claim_input_ids, attention_mask=claim_attention_mask).last_hidden_state[:,0,:]
        analyse_embedding = self.bert_t(input_ids=analyse_input_ids, attention_mask=analyse_attention_mask).last_hidden_state[:,0,:]
        true_embedding = self.bert_r(input_ids=true_input_ids, attention_mask=true_attention_mask).last_hidden_state[:,0,:]
        false_embedding = self.bert_r(input_ids=false_input_ids, attention_mask=false_attention_mask).last_hidden_state[:,0,:]


        # Attention weighted adjustment
        # true_adjusted, _ = self.attention_w(query=claim_embedding, key=true_embedding, value=true_embedding)
        w_t = torch.sigmoid(self.MLP(torch.cat([claim_embedding, true_embedding], dim=1)))
        u_t = torch.zeros_like(w_t)

        # false_adjusted, _ = self.attention_w(query=claim_embedding, key=false_embedding, value=false_embedding)
        w_f = torch.sigmoid(self.MLP(torch.cat([claim_embedding, false_embedding], dim=1)))


        # true_temp, _ = self.true_att(query=claim_embedding, key=true_embedding, value=true_embedding)
        # true_output = w_t*true_temp
        # false_temp, _ = self.false_att(query=claim_embedding, key=false_embedding, value=false_embedding)
        # false_output = w_f*false_temp
        # analyse_output, _ = self.analyse_att(query=claim_embedding, key=analyse_embedding, value=analyse_embedding)
        true_output = true_embedding
        false_output = false_embedding
        analyse_output = analyse_embedding

        claim_true_embedding = torch.cat([claim_embedding, true_output], dim=1)
        claim_false_embedding = torch.cat([claim_embedding, false_output], dim=1)
        claim_analyse_embedding = torch.cat([claim_embedding, analyse_output], dim=1)
        # claim_true_embedding = claim_embedding+true_output
        # claim_false_embedding = claim_embedding+false_output
        # claim_analyse_embedding = claim_embedding+analyse_output
        # claim_true_embedding = self.transformer1(claim_true_embedding)
        # claim_true_embedding = self.transformer2(claim_true_embedding)
        # claim_false_embedding = self.transformer1(claim_false_embedding)
        # claim_false_embedding = self.transformer2(claim_false_embedding)
        # claim_analyse_embedding = self.transformer1(claim_analyse_embedding)
        # claim_analyse_embedding = self.transformer2(claim_analyse_embedding)
        # claim_true_embedding = self.transformer_t(w_t*claim_true_embedding)
        # claim_false_embedding = self.transformer_f(w_f*claim_false_embedding)
        claim_true_embedding = self.transformer(w_t*claim_true_embedding)
        claim_false_embedding = self.transformer(w_f*claim_false_embedding)
        claim_analyse_embedding = self.transformer_c(claim_analyse_embedding)

        # Step 5: Combine and classify
        combined = torch.stack([claim_analyse_embedding, claim_true_embedding, claim_false_embedding], dim=1)
        # combined = self.transformer1(combined)
        # combined = self.transformer2(combined)
        attention_output, _ = self.attention(combined, combined, combined)
        output = torch.flatten(attention_output, start_dim=1)
        # output = torch.cat([claim_embedding, torch.flatten(attention_output, start_dim=1)], dim=1)
        # output = torch.cat([claim_embedding ,torch.cat([attention_output[:, i, :] for i in range(3)], dim=-1)],dim=1) # Apply mean pooling across all outputs
        logits = self.fc3(self.relu(self.fc2(self.fc1(output))))
        return logits, w_t, w_f
train_file = "snopes/train_set.tsv"
val_file = "snopes/val_set.tsv"
test_file = "snopes/test_set.tsv"

# train_file = "poli/train_set.tsv"
# val_file = "poli/val_set.tsv"
# test_file = "poli/test_set.tsv"
# 检查数据集是否存在
if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
    # 如果文件存在，加载数据
    train_data = pd.read_csv(train_file, sep='\t')
    val_data = pd.read_csv(val_file, sep='\t')
    test_data = pd.read_csv(test_file, sep='\t')
    print("数据集文件已存在，直接加载。")
else:
    # 如果文件不存在，随机划分数据集
    print("数据集文件不存在，开始随机划分数据集。")
    # 假设原始数据文件
    original_file = "tokenization_snopes.tsv"
    # original_file = "poli_tokenization.tsv"
    df = pd.read_csv(original_file, sep='\t')
    df.columns = ['claim', 'analyse1', 'analyse2', 'analyse3', 'analyse4', 
                  'flipped_claim', 'flipped_analyse1', 'flipped_analyse2', 'flipped_analyse3', 'flipped_analyse4', 
                  'reason_true', 'reason_false', 'label']
    
    # 数据划分
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=7)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=7)

    # 保存到文件
    train_data.to_csv(train_file, sep='\t', index=False)
    val_data.to_csv(val_file, sep='\t', index=False)
    test_data.to_csv(test_file, sep='\t', index=False)
    print("数据集划分完成，并已保存到文件。")

# 检查GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型和tokenizer
model = SemanticAttentionClassifier().to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def extract_columns(data):
    claims = data['claim'].tolist()
    analyses = data[['analyse1', 'analyse2', 'analyse3', 'analyse4']].values.tolist()
    reasons_true = data['reason_true'].tolist()
    reasons_false = data['reason_false'].tolist()
    labels = data['label'].tolist()
    return claims, analyses, reasons_true, reasons_false, labels

train_claims, train_analyses, train_reasons_true, train_reasons_false, train_labels = extract_columns(train_data)
val_claims, val_analyses, val_reasons_true, val_reasons_false, val_labels = extract_columns(val_data)
test_claims, test_analyses, test_reasons_true, test_reasons_false, test_labels = extract_columns(test_data)


# 创建数据集和数据加载器
train_dataset = ClaimDataset(train_claims, train_analyses, train_reasons_true, train_reasons_false, train_labels, tokenizer, max_length=128)
val_dataset = ClaimDataset(val_claims, val_analyses, val_reasons_true, val_reasons_false, val_labels, tokenizer, max_length=128)
test_dataset = ClaimDataset(test_claims, test_analyses, test_reasons_true, test_reasons_false, test_labels, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 训练设置
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
loss_tf = nn.BCEWithLogitsLoss()
epochs = 10
best_accuracy = 0.0
model_save_path = 'snopes/best_model.pth'
# model_save_path = 'poli/best_model.pth'
# 训练循环
preds, true_labels = [], []
import itertools
alpha_range = [0.1, 0.05]  # 正样本损失权重
beta_range = [0.1, 0.05]   # 负样本损失权重
param_grid = list(itertools.product(alpha_range, beta_range))  # 生成所有 (alpha, beta) 组合
# 初始化最佳参数和性能
best_val_accuracy = 0.0
best_params = None
# for alpha, beta in param_grid:
    # print(f"Testing combination: alpha={alpha}, beta={beta}")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0  
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
        # 获取输入数据并移动到设备
        # claim_input = batch['claim'].to(device)
        # analyse_input = batch['analyse'].to(device)
        # true_input = batch['true'].to(device)
        # false_input = batch['false'].to(device)
        labels = batch['label'].to(device)
        claim_input_ids = batch['claim_input_ids'].to(device)
        claim_attention_mask = batch['claim_attention_mask'].to(device)
        analyse_input_ids = batch['analyse_input_ids'].to(device)
        analyse_attention_mask = batch['analyse_attention_mask'].to(device)
        true_input_ids = batch['true_input_ids'].to(device)
        true_attention_mask = batch['true_attention_mask'].to(device)
        false_input_ids = batch['false_input_ids'].to(device)
        false_attention_mask = batch['false_attention_mask'].to(device)
        logits, w_t, w_f = model(
            claim_input_ids=claim_input_ids, 
            claim_attention_mask=claim_attention_mask,
            analyse_input_ids=analyse_input_ids, 
            analyse_attention_mask=analyse_attention_mask,
            true_input_ids=true_input_ids, 
            true_attention_mask=true_attention_mask,
            false_input_ids=false_input_ids, 
            false_attention_mask=false_attention_mask,
            labels=labels
        )
        
        # 计算损失
        # loss = (
        #     loss_fn(logits, labels)
        #     + 0.1 * loss_tf(w_t, labels.view(-1, 1).float())
        #     + 0.1 * loss_tf(w_f, 1 - labels.view(-1, 1).float())
        # )
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(logits, dim=1)
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        running_loss += loss.item()

    # 输出损失值
    accuracy, precision, recall, f1_binary, f1_macro, f1_micro, f1_weighted = compute_metrics(preds, true_labels)
    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss after Epoch {epoch+1}: {avg_loss:.4f}")
    print(f"train Accuracy: {accuracy:.4f}")
    print(f"train Precision: {precision:.4f}")
    print(f"train Recall: {recall:.4f}")
    print(f"train F1 Binary: {f1_binary:.4f}")
    print(f"train F1 Macro: {f1_macro:.4f}")
    print(f"train F1 Micro: {f1_micro:.4f}")
    print(f"train F1 Weighted: {f1_weighted:.4f}")
    # best_accuracy = save_best_model(model, accuracy, best_accuracy, epoch, model_save_path)
    # 验证集评估
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch['label'].to(device)
            claim_input_ids = batch['claim_input_ids'].to(device)
            claim_attention_mask = batch['claim_attention_mask'].to(device)
            analyse_input_ids = batch['analyse_input_ids'].to(device)
            analyse_attention_mask = batch['analyse_attention_mask'].to(device)
            true_input_ids = batch['true_input_ids'].to(device)
            true_attention_mask = batch['true_attention_mask'].to(device)
            false_input_ids = batch['false_input_ids'].to(device)
            false_attention_mask = batch['false_attention_mask'].to(device)
            logits, _ ,_ = model(
                claim_input_ids=claim_input_ids, 
                claim_attention_mask=claim_attention_mask,
                analyse_input_ids=analyse_input_ids, 
                analyse_attention_mask=analyse_attention_mask,
                true_input_ids=true_input_ids, 
                true_attention_mask=true_attention_mask,
                false_input_ids=false_input_ids, 
                false_attention_mask=false_attention_mask,
                labels=labels
            )
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算并输出评估指标
    accuracy, precision, recall, f1_binary, f1_macro, f1_micro, f1_weighted = compute_metrics(preds, true_labels)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Binary: {f1_binary:.4f}")
    print(f"Validation F1 Macro: {f1_macro:.4f}")
    print(f"Validation F1 Micro: {f1_micro:.4f}")
    print(f"Validation F1 Weighted: {f1_weighted:.4f}")
    # if accuracy > best_val_accuracy:
    #     best_val_accuracy = accuracy
    #     best_params = (alpha, beta)
    #     print(f"New Best Parameters: alpha={alpha}, beta={beta}, Validation Accuracy={accuracy:.4f}")
        # 保存当前最优模型
    # 保存最好的模型
    best_accuracy = save_best_model(model, accuracy, best_accuracy, epoch, model_save_path)
# print(f"Best Parameters: alpha={best_params[0]}, beta={best_params[1]}")
# print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
# 加载最好的模型并在测试集上进行测试
model.load_state_dict(torch.load(model_save_path))
model.eval()
test_preds, test_true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        labels = batch['label'].to(device)
        claim_input_ids = batch['claim_input_ids'].to(device)
        claim_attention_mask = batch['claim_attention_mask'].to(device)
        analyse_input_ids = batch['analyse_input_ids'].to(device)
        analyse_attention_mask = batch['analyse_attention_mask'].to(device)
        true_input_ids = batch['true_input_ids'].to(device)
        true_attention_mask = batch['true_attention_mask'].to(device)
        false_input_ids = batch['false_input_ids'].to(device)
        false_attention_mask = batch['false_attention_mask'].to(device)
        logits, _ ,_ = model(
            claim_input_ids=claim_input_ids, 
            claim_attention_mask=claim_attention_mask,
            analyse_input_ids=analyse_input_ids, 
            analyse_attention_mask=analyse_attention_mask,
            true_input_ids=true_input_ids, 
            true_attention_mask=true_attention_mask,
            false_input_ids=false_input_ids, 
            false_attention_mask=false_attention_mask,
            labels=labels
        )
        predictions = torch.argmax(logits, dim=1)

        test_preds.extend(predictions.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

# 计算测试集的性能指标
accuracy, precision, recall, f1_binary, f1_macro, f1_micro, f1_weighted = compute_metrics(test_preds, test_true_labels)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Binary: {f1_binary:.4f}")
print(f"Test F1 Macro: {f1_macro:.4f}")
print(f"Test F1 Micro: {f1_micro:.4f}")
print(f"Test F1 Weighted: {f1_weighted:.4f}")
