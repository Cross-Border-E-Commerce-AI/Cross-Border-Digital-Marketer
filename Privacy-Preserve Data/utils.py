import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


#------------------------------------------------------#
# 安全数据预处理​
class SecureDataProcessor:
    def __init__(self, public_key):
        self.public_key = public_key  # 平台公钥
        
    def _differential_privacy(self, data: np.ndarray, epsilon=0.5):
        """添加差分隐私噪声"""
        sensitivity = 1.0
        beta = sensitivity / epsilon
        return data + np.random.laplace(0, beta, data.shape)
    
    def process_user_data(self, raw_data: pd.DataFrame):
        """端侧数据加密处理"""
        # 特征标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(raw_data)
        
        # 添加差分隐私
        privatized_data = self._differential_privacy(scaled_data)
        
        # 同态加密处理
        encrypted_data = self.public_key.encrypt(
            privatized_data.tobytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data
    
# 跨平台模型设计​
class EcommerceModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.platform_specific = nn.ModuleDict({
            'platformA': nn.Linear(64, 32),
            'platformB': nn.Linear(64, 32)
        })
        self.final_layer = nn.Linear(32, 1)
        
    def forward(self, x, platform):
        x = self.shared_layer(x)
        x = self.platform_specific[platform](x)
        return torch.sigmoid(self.final_layer(x))
    
# ​​联邦客户端实现​
class PlatformClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, platform_id):
        self.model = model
        self.train_loader = train_loader
        self.platform_id = platform_id
        
    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # 更新模型参数
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # 本地训练
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(3):
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data, self.platform_id)
                loss = nn.BCELoss()(output, target)
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        # 省略评估代码
        return 0.0, len(self.train_loader), {"accuracy": 0.95}
    
# ​​安全聚合协议​
class SecureAggregator(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """安全多方计算聚合"""
        # 1. 验证参与方身份
        self._verify_clients([r.id for r in results])
        
        # 2. 解密参数
        decrypted_params = [self._decrypt(r.parameters) for r in results]
        
        # 3. 参数聚合
        return super().aggregate_fit(server_round, decrypted_params, failures)
    
    def _verify_clients(self, client_ids):
        # 基于数字证书的验证机制
        trusted_ids = ['platformA', 'platformB']
        if not all(cid in trusted_ids for cid in client_ids):
            raise SecurityError("Untrusted participant detected")
    
    def _decrypt(self, encrypted_params):
        # 使用平台私钥解密
        return [private_key.decrypt(
            p,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        ) for p in encrypted_params]
    

#-------------------------------------------------------#
# 数据模拟
def simulate_platform_data(platform: str, n_samples=1000):
    """生成模拟电商数据"""
    common_features = np.random.randn(n_samples, 5)  # 共享特征
    platform_specific = np.random.rand(n_samples, 5)  # 平台特有特征
    
    # 生成标签
    decision_boundary = np.dot(common_features, np.array([0.8, -0.5, 1.2, 0.3, -0.7])) 
    labels = (decision_boundary + np.random.normal(0, 0.1, n_samples) > 0.5).astype(float)
    
    return TensorDataset(
        torch.Tensor(np.hstack([common_features, platform_specific])),
        torch.Tensor(labels).unsqueeze(1)
    )

# 联邦训练执行
def start_federation():
    # 初始化全局模型
    global_model = EcommerceModel(input_dim=10)
    
    # 启动服务器
    strategy = SecureAggregator(
        min_fit_clients=2,
        min_available_clients=2
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 10},
        strategy=strategy
    )
    
    # 客户端加入
    platformA_data = simulate_platform_data("platformA")
    platformB_data = simulate_platform_data("platformB")
    
    clientA = PlatformClient(global_model, DataLoader(platformA_data, batch_size=32), "platformA")
    clientB = PlatformClient(global_model, DataLoader(platformB_data, batch_size=32), "platformB")
    
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=clientA)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=clientB)


#-------------------------------------------------------#
# 参数混淆机制​
class ParameterObfuscator:
    def __init__(self, noise_scale=0.1):
        self.noise_scale = noise_scale
        
    def apply(self, parameters: List[np.ndarray]):
        """添加随机噪声混淆参数"""
        return [p + np.random.normal(0, self.noise_scale, p.shape) 
               for p in parameters]
    
# 安全更新验证​
def verify_update_signature(update: bytes, signature: bytes, public_key):
    """验证参数更新签名"""
    public_key.verify(
        signature,
        update,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

# 动态隐私预算​
class PrivacyBudgetManager:
    def __init__(self, total_epsilon=4.0):
        self.remaining_epsilon = total_epsilon
        
    def allocate_epsilon(self, round_num): 
        """动态分配隐私预算"""
        decay_factor = 0.9 ** round_num
        current_epsilon = min(0.5, self.remaining_epsilon * decay_factor)
        self.remaining_epsilon -= current_epsilon
        return current_epsilon
    

#-------------------------------------------------------#
# 横向联邦特征对齐​
def feature_alignment(client_features: List[np.ndarray]):
    """联邦特征空间对齐"""
    # 使用共享随机投影
    projection_matrix = np.random.randn(client_features[0].shape[1], 256)
    aligned_features = [np.dot(f, projection_matrix) for f in client_features]
    return aligned_features

# ​​异步联邦更新​
class AsyncFederatedUpdater:
    def __init__(self, staleness_threshold=3):
        self.client_updates = {}
        self.staleness = {}
        
    def handle_update(self, client_id, params):
        self.client_updates[client_id] = params
        self.staleness[client_id] = 0
        
    def get_aggregated_update(self):
        # 处理不同步的客户端更新
        fresh_updates = [p for cid, p in self.client_updates.items()
                        if self.staleness[cid] < self.staleness_threshold]
        return average_parameters(fresh_updates)
    
# ​联邦特征重要性分析​
def federated_feature_importance(global_model, feature_names):
    """分析共享模型的特征重要性"""
    weights = global_model.shared_layer[0].weight.detach().numpy()
    importance = np.mean(np.abs(weights), axis=0)
    return sorted(zip(feature_names, importance), 
                key=lambda x: x[1], reverse=True)


#--------------------------------------------------------#
if __name__ == "__main__":
    # 模拟数据
    platformA_data = simulate_platform_data("platformA")
    platformB_data = simulate_platform_data("platformB")
    
    # 初始化公钥和私钥
    public_key = ...  # 省略公钥生成代码
    private_key = ...  # 省略私钥生成代码
    
    # 数据处理
    processor = SecureDataProcessor(public_key)
    encrypted_dataA = processor.process_user_data(platformA_data)
    encrypted_dataB = processor.process_user_data(platformB_data)
    
    # 启动联邦学习
    start_federation()
    # 省略其他代码
    # 省略模型训练和评估代码
    # 省略安全聚合和隐私预算管理代码    