import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, Beta
from torch.nn.functional import relu




def np2torch(x: np.ndarray,
             dtype: torch.dtype = torch.float32,
             device: torch.device = torch.device('cpu')):
    return torch.from_numpy(x).type(dtype).to(device)

def build_mlp(in_dim: int,
              h_dims: list,
              h_actv: nn.Module):
    layers = []
    for i in range(len(h_dims)):
        if i == 0:
            layers.append(nn.Linear(in_dim, h_dims[i]))
        else:
            layers.append(nn.Linear(h_dims[i-1], h_dims[i]))
        layers.append(h_actv)
        layers.append(nn.BatchNorm1d(h_dims[i]))
    return nn.Sequential(*layers)

def calc_gae(reward_batch: torch.Tensor,
             val_batch: torch.Tensor,
             last_val,
             gamma: float,
             lmbda: float):
    advantage_batch = np.zeros(shape=(reward_batch.shape[0], 1), dtype=np.float32)
    
    advantage = 0.0
    for idx in reversed(range(reward_batch.shape[0])):
        if idx == reward_batch.shape[0]-1:
            next_val = last_val
        else:
            next_val = val_batch[idx+1]
        delta = reward_batch[idx] + gamma * next_val - val_batch[idx]
        advantage = delta + gamma * lmbda * advantage
        advantage_batch[idx] = advantage
    return_batch = advantage_batch + val_batch
    return return_batch, advantage_batch

class PPOBufferClass:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 buffer_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size
        
        self.obs_buffer = np.zeros(shape=(buffer_size, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros(shape=(buffer_size, act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.log_prob_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.return_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.advantage_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.val_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)

        self.random_generator = np.random.default_rng()
        self.start_idx, self.pointer = 0, 0
        
    def put(self,
            obs,
            act,
            reward,
            val,
            log_prob):
        self.obs_buffer[self.pointer] = obs
        self.act_buffer[self.pointer] = act
        self.reward_buffer[self.pointer] = reward
        self.val_buffer[self.pointer] = val
        self.log_prob_buffer[self.pointer] = log_prob
        self.pointer += 1
    
    def get_gae_batch(self,
                      gamma,
                      lmbda,
                      last_val):
        path_slice = slice(self.start_idx, self.pointer)
        val_mini_buffer = self.val_buffer[path_slice]
        
        self.return_buffer[path_slice], self.advantage_buffer[path_slice] = calc_gae(
                                                                                    self.reward_buffer[path_slice],
                                                                                    val_mini_buffer,
                                                                                    last_val,
                                                                                    gamma,
                                                                                    lmbda
                                                                                    )
        self.start_idx = self.pointer
    
    def get_mini_batch(self,
                       mini_batch_size):
        assert mini_batch_size <= self.pointer
        indices = np.arange(self.pointer)
        self.random_generator.shuffle(indices)
        
        split_indices = []
        point = mini_batch_size
        while point < self.pointer:
            split_indices.append(point)
            point += mini_batch_size
        
        temp_data = {
                    'obs': np.split(self.obs_buffer[indices], split_indices),
                    'act': np.split(self.act_buffer[indices], split_indices),
                    'reward': np.split(self.reward_buffer[indices], split_indices),
                    'val': np.split(self.val_buffer[indices], split_indices),
                    'log_prob': np.split(self.log_prob_buffer[indices], split_indices),
                    'return': np.split(self.return_buffer[indices], split_indices),
                    'advantage': np.split(self.advantage_buffer[indices], split_indices)
                    }
        
        data = []
        for k in range(len(temp_data['obs'])):
            data.append({
                        'obs': temp_data['obs'][k],
                        'action': temp_data['act'][k],
                        'reward': temp_data['reward'][k],
                        'val': temp_data['val'][k],
                        'log_prob': temp_data['log_prob'][k],
                        'return': temp_data['return'][k],
                        'advantage': temp_data['advantage'][k]
                        })
        return data

    def clear(self):
        self.start_idx, self.pointer = 0, 0
    
class ActorClass(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 h_dims: list,
                 act_dim: int,
                 h_actv: nn.Module,
                 mu_actv: nn.Module,
                 lr_actor: float,
                 beta_dist: bool = False):
        super(ActorClass, self).__init__()
        self.beta_dist = beta_dist
        self.layers = build_mlp(in_dim=obs_dim, h_dims=h_dims, h_actv=h_actv)
        if beta_dist:
            self.mu_head = nn.Linear(h_dims[-1], act_dim*2)
        else:
            self.mu_head = nn.Linear(h_dims[-1], act_dim)
        self.mu_actv = mu_actv
        # self.apply(self._init_weights)

    def forward(self,
                obs: torch.Tensor):
        x = self.layers(obs)
        if self.mu_actv is not None:
            mu = self.mu_actv(self.mu_head(x))
        else:
            mu = self.mu_head(x)

        if self.beta_dist:
            alpha, beta = torch.split(mu, mu.shape[-1]//2, dim=-1)
            alpha = F.softplus(alpha)
            beta = F.softplus(beta)
            return alpha, beta
        else:
            return mu

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
class CriticClass(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 h_dims: list,
                 val_dim: int,
                 h_actv: nn.Module,
                 out_actv: nn.Module,
                 lr_critic: float):
        super(CriticClass, self).__init__()
        self.layers = build_mlp(in_dim=obs_dim, h_dims=h_dims, h_actv=h_actv)
        self.val_head = nn.Linear(h_dims[-1], val_dim)
        self.out_actv = out_actv
        if self.out_actv is not None:
            self.out_actv = out_actv
        # self.apply(self._init_weights)
    
    def forward(self,
                obs):
        x = self.layers(obs)
        val = self.val_head(x)
        if self.out_actv is not None:
            val = self.out_actv(val)
        return val

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
class PPOClass(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 h_dims: list,
                 gamma: float,
                 lmbda: float,
                 lr_actorcritic: float,
                 clip_ratio: float,
                 value_coef: float,
                 entropy_coef: float,
                 max_grad: float,
                 beta_dist: bool = False,
                 ):
        super(PPOClass, self).__init__()
        
        # self.max_pos = max_pos
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef   
        self.entropy_coef = entropy_coef
        self.max_grad = max_grad
        self.beta_dist = beta_dist
        if beta_dist:
            self.actor = ActorClass(obs_dim=obs_dim,h_dims=h_dims,act_dim=act_dim,h_actv=nn.ReLU(),mu_actv=nn.Tanh(),lr_actor=lr_actorcritic, beta_dist=beta_dist)
        else:
            self.actor = ActorClass(obs_dim=obs_dim,h_dims=h_dims,act_dim=act_dim,h_actv=nn.ReLU(),mu_actv=nn.Tanh(),lr_actor=lr_actorcritic)
            self.log_std = nn.Parameter(torch.ones(act_dim) * torch.log(torch.tensor((1.0))), requires_grad=True)
        self.critic = CriticClass(obs_dim=obs_dim,h_dims=h_dims,val_dim=1,h_actv=nn.ReLU(),out_actv=None,lr_critic=lr_actorcritic)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actorcritic)
    
    def forward(self,
                obs: torch.Tensor):
        if self.beta_dist:
            alpha, beta = self.actor(obs)
            dist = Beta(alpha, beta)
        else:
            mu = self.actor(obs)
            # mu = torch.clamp(mu, -0.16, 0.16)
            dist = Normal(mu, torch.exp(self.log_std))
        val = self.critic(obs)
        return dist, val
    
    def get_action(self,
                   obs):
        obs_torch = torch.unsqueeze(torch.FloatTensor(obs), 0)
        self.actor.eval()
        self.critic.eval()
        dist, val = self.forward(obs_torch)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=-1)
        return action[0].detach().numpy(), torch.squeeze(log_prob).detach().numpy(), torch.squeeze(val).detach().numpy()
    
    def get_best_action(self, obs):
        obs_torch = torch.unsqueeze(torch.FloatTensor(obs), 0)
        self.actor.eval()
        self.critic.eval()
        dist, val = self.forward(obs_torch)
        best_action = dist.mean()
        return best_action.detach().numpy()

        
    def get_val(self,
                  obs):
        obs_torch = torch.unsqueeze(torch.FloatTensor(obs), 0)
        dist, val = self.forward(obs_torch)
        return torch.squeeze(val).detach().numpy()

    def eval_action(self,
                    obs_batch,
                    act_batch):
        obs_torch = obs_batch.clone().detach()
        action_torch = act_batch.clone().detach()
        dist, val = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        # entropy = dist.entropy # truncated
        entropy = dist.entropy() #Normal
        return log_prob, val, entropy
    
    def update(self,
               obs_batch,
               act_batch,
               log_prob_batch,
               advantage_batch,
               return_batch):
        self.actor.train()
        self.critic.train()
        new_log_prob_batch, val_batch, entropy = self.eval_action(obs_batch, act_batch)
        ratio = torch.exp(new_log_prob_batch - log_prob_batch)
        
        surr1 = ratio * advantage_batch
        surr2 = torch.clip(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * advantage_batch
        actor_loss = -torch.mean(torch.min(surr1, surr2)) + self.entropy_coef*entropy.mean()
        critic_loss = self.value_coef * torch.mean((val_batch-return_batch)**2)        
        total_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # nn.utils.clip_grad_value_(self.log_std, 1.0)
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad)
        self.optimizer.step()

        return actor_loss.detach(), critic_loss.detach(), total_loss.detach()
    
if __name__ == "__main__":
    ppo = PPOClass( 
                    obs_dim=3,
                    act_dim=1,
                    h_dims=[128, 128],
                    gamma=0.99,
                    lmbda=0.95,
                    lr_actorcritic=1e-5,
                    clip_ratio=0.2,
                    value_coef=0.5,
                    entropy_coef=0.01,
                    max_grad=0.5,
                    beta_dist=True
                    )
    # torch.save(ppo.state_dict(), "ppo_model.pth")
    # ppo = torch.load("ppo_model.pth")
    ppo.load_state_dict(torch.load("ppo_model.pth"))
    print(ppo)
    print(ppo.beta_dist)