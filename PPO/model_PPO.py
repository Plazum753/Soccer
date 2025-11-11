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
            if i%5==0 :
                layer_actor.append(nn.LayerNorm(hidden_size))
            layer_actor.append(nn.ReLU())
        layer_actor.append(nn.Linear(hidden_size,n_pions+4))
        
        self.actor = nn.Sequential(*layer_actor)
        
        layer_critic = []
        layer_critic.append(nn.Linear(input_size, hidden_size))
        layer_critic.append(nn.ReLU())
        for i in range(1,n_hidden_layers):
            layer_critic.append(nn.Linear(hidden_size, hidden_size))
            if i%5==0 :
                layer_critic.append(nn.LayerNorm(hidden_size))
            layer_critic.append(nn.ReLU())
        layer_critic.append(nn.Linear(hidden_size,1))
        
        self.critic = nn.Sequential(*layer_critic)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = self.lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = self.lr)
    
    def forward(self, state, n_pions=5, action=None) :
        out_actor = self.actor(state)
        valeur = self.critic(state)

        if out_actor.dim() == 1 :
            out_actor = out_actor.unsqueeze(0)
                    
        pions = out_actor[:, :n_pions]
        angle = out_actor[:, n_pions:n_pions+2]
        vitesse = out_actor[:, -2:]
        
        pions_prob = dist.Categorical(logits=pions)
        pion = pions_prob.sample() if action == None else action[0] 
        log_prob_pion = pions_prob.log_prob(pion)
        
        sigma_angle = F.softplus(angle[:, 1]) + 1e-6 # empêche l'angle d'être négatif ou trop proche de 0
        normal_angle = dist.Normal(angle[:, 0], sigma_angle)
        epsilon_angle = torch.randn_like(sigma_angle) 
        angle_choix = angle[:, 0] + sigma_angle * epsilon_angle if action == None else action[1] 
        log_prob_angle = normal_angle.log_prob(angle_choix)
        
        sigma_vitesse = F.softplus(vitesse[:, 1]) + 1e-6
        normal_vitesse = dist.Normal(vitesse[:, 0], sigma_vitesse)
        epsilon_vitesse = torch.randn_like(sigma_vitesse)
        vitesse_choix = vitesse[:, 0] + sigma_vitesse * epsilon_vitesse if action == None else action[2] 
        log_prob_vitesse = normal_vitesse.log_prob(vitesse_choix)
        vitesse_choix = 19 * torch.sigmoid(vitesse_choix) + 1
        
        log_proba = log_prob_pion + log_prob_angle + log_prob_vitesse
        
        action = (pion, angle_choix, vitesse_choix)
        
        return (action, log_proba, valeur)
        
class Trainer :
    def __init__(self, model, batch_size=64, epoch=4, epsilon=0.2):
        self.model = model
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_step(self, batch):
        idx = torch.randperm(self.batch_size)

        states = torch.stack([batch[i][0] for i in idx]).to(self.device)
        valeurs = torch.stack([batch[i][2] for i in idx]).detach().to(self.device)
        log_probas = torch.stack([batch[i][3] for i in idx]).detach().to(self.device)
        rewards = torch.tensor([batch[i][4] for i in idx], dtype=torch.float32, device=self.device)
        
        pions = torch.stack([batch[i][1][0] for i in idx]).to(self.device).detach()
        angles = torch.stack([batch[i][1][1] for i in idx]).to(self.device).detach()
        vitesses = torch.stack([batch[i][1][2] for i in idx]).to(self.device).detach()
        actions = (pions, angles, vitesses)
        
        with torch.no_grad() :
            avantages = (rewards - valeurs).detach()
        
        for i in range(self.epoch):
            _, _, valeur_pred = self.model.forward(states, action=actions)
            
            loss_critic = self.loss_criterion(valeur_pred.squeeze(), rewards)
            
            self.model.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.model.optimizer_critic.step()
            
            _, log_proba_new, _ = self.model.forward(states, action=actions)

            ratio = torch.exp(log_proba_new - log_probas)
            loss_actor = -(torch.min(ratio * avantages,torch.clamp(ratio,1-self.epsilon, 1+self.epsilon) * avantages)).mean()
            
            self.model.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.model.optimizer_actor.step()