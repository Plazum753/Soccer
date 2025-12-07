import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

class Model :
    def __init__(self, input_size, hidden_size, n_hidden_layers, n_pions=5, lr=1e-4) :
        self.lr = lr
        
        layer_actor = []
        layer_actor.append(nn.Linear(input_size, hidden_size))
        layer_actor.append(nn.ReLU())
        for i in range(1, n_hidden_layers):
            layer_actor.append(nn.Linear(hidden_size, hidden_size))
            layer_actor.append(nn.ReLU())
        layer_actor.append(nn.Linear(hidden_size,n_pions+6))
        
        self.actor = nn.Sequential(*layer_actor)

        layer_critic = []
        layer_critic.append(nn.Linear(input_size, hidden_size))
        layer_critic.append(nn.ReLU())
        for i in range(1,n_hidden_layers):
            layer_critic.append(nn.Linear(hidden_size, hidden_size))
            layer_critic.append(nn.ReLU())
        layer_critic.append(nn.Linear(hidden_size,1))
        
        self.critic = nn.Sequential(*layer_critic)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr = self.lr)

    
    def forward(self, state, n_pions=5, action=None, training=True) :
        out_actor = self.actor(state)
        valeur = self.critic(state)

        if out_actor.dim() == 1 :
            out_actor = out_actor.unsqueeze(0)
        
        pions = out_actor[:, :n_pions]
        dx = out_actor[:, -6:-4]  
        dy = out_actor[:, -4:-2]
        v = out_actor[:, -2:]
        
        pions = torch.nan_to_num(pions, nan=0.0, posinf=1e3, neginf=-1e3)
        dx = torch.nan_to_num(dx, nan=0.0, posinf=1e3, neginf=-1e3)
        dy = torch.nan_to_num(dy, nan=0.0, posinf=1e3, neginf=-1e3)
        v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
        
        pions_prob = dist.Categorical(logits=pions)
        pion = pions_prob.sample() if action is None else action[0] 
        log_prob_pion = pions_prob.log_prob(pion)
        entropy_pions = pions_prob.entropy()
        
        sigma_dx = F.softplus(dx[:, 1]) + 1e-6
        normal_dx = dist.Normal(dx[:, 0], sigma_dx)
        epsilon_dx = torch.randn_like(sigma_dx)
        dx_choix = dx[:, 0] + sigma_dx * epsilon_dx if action is None else action[1]
        log_prob_dx = normal_dx.log_prob(dx_choix)
        
        sigma_dy = F.softplus(dy[:, 1]) + 1e-6
        normal_dy = dist.Normal(dy[:, 0], sigma_dy)
        epsilon_dy = torch.randn_like(sigma_dy)
        dy_choix = dy[:, 0] + sigma_dy * epsilon_dy if action is None else action[2]
        log_prob_dy = normal_dy.log_prob(dy_choix)
        
        sigma_v = F.softplus(v[:, 1]) + 1e-6
        normal_v = dist.Normal(v[:, 0], sigma_v)
        epsilon_v = torch.randn_like(sigma_v)
        v_choix = v[:, 0] + sigma_v * epsilon_v if action is None else action[3]
        log_prob_v = normal_v.log_prob(v_choix)
        
        if training == False :
            pion = torch.argmax(pions, dim=1)
            dx_choix = dx[:, 0]
            dy_choix = dy[:, 0]
            v_choix = v[:, 0]
                
        entropy_pions = pions_prob.entropy() 
        entropy_dx = normal_dx.entropy()
        entropy_dy = normal_dy.entropy()
        entropy_v = normal_v.entropy()
        
        entropie = entropy_pions + entropy_dx + entropy_dy + entropy_v
        
        log_proba = log_prob_pion + log_prob_dx + log_prob_dy + log_prob_v
        
        action = (pion, dx_choix, dy_choix, v_choix)
        #print("action :",action)
        
        return (action, log_proba, valeur, entropie)
        
class Trainer :
    def __init__(self, model, batch_size=256, epoch=4, epsilon=0.3, n_pions=5):
        self.model = model
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_pions = n_pions
    
    def train_step(self, batch):
        idx = torch.randperm(self.batch_size)

        states = torch.stack([batch[i][0] for i in idx]).to(self.device)
        avantages = torch.tensor([batch[i][2] for i in idx], dtype=torch.float32, device=self.device)
        log_probas = torch.stack([batch[i][3] for i in idx]).detach().to(self.device)
        returns = torch.tensor([batch[i][-1] for i in idx], dtype=torch.float32, device=self.device)
        
        pions = torch.stack([batch[i][1][0] for i in idx]).to(self.device).detach()
        dx = torch.stack([batch[i][1][1] for i in idx]).to(self.device).detach()
        dy = torch.stack([batch[i][1][2] for i in idx]).to(self.device).detach()
        v = torch.stack([batch[i][1][3] for i in idx]).to(self.device).detach()
        actions = (pions, dx, dy, v)
        
        for i in range(self.epoch):
            _, log_proba_new, valeur_pred, entropie = self.model.forward(states, action=actions, n_pions=self.n_pions)
            
            ratio = torch.exp(log_proba_new - log_probas)
            L_clip = -torch.min(ratio * avantages,torch.clamp(ratio,1-self.epsilon, 1+self.epsilon) * avantages)
            
            L_vf = self.loss_criterion(valeur_pred.squeeze(), returns)

            S = entropie.mean()
            
            loss = L_clip.mean() + 0.5 * L_vf - 0.01 * S
            
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()