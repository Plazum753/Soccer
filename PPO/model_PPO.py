import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

class Model :
    def __init__(self, input_size, hidden_size, n_pions=5) :
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_pions+4)
            )
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor.to(self.device)
        self.critic.to(self.device)
    
    def forward(self, state, n_pions=5, action=None) :
        out_actor = self.actor(state)
        valeur = self.critic(state)
        
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
        
        log_proba = log_prob_pion + log_prob_angle + log_prob_vitesse
        
        action = (pion, angle_choix, vitesse_choix)
        
        return (action, log_proba, valeur)
    
class Trainer :
    def __init__(self, model, lr=1e-4, batch_size=64, epoch=4, epsilon=0.2): #TODO
        self.lr = lr
        self.model = model
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.optimizer_critic = optim.Adam(self.model.critic.parameters(), lr = self.lr)
        self.optimizer_actor = optim.Adam(self.model.actor.parameters(), lr = self.lr)
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_step(self, batch): # batch = (states, actions, rewards, log_probas, valeurs)
        idx = torch.randperm(self.batch_size)
        
        states = torch.tensor([batch[i][0] for i in idx], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([batch[i][2] for i in idx], dtype=torch.float32, device=self.device)
        log_probas = torch.tensor([batch[i][3] for i in idx], dtype=torch.float32, device=self.device)
        valeurs = torch.tensor([batch[i][4] for i in idx], dtype=torch.float32, device=self.device)
        
        pions = torch.tensor([batch[i][1][0] for i in idx], dtype=torch.float32, device=self.device)
        angles = torch.tensor([batch[i][1][1] for i in idx], dtype=torch.float32, device=self.device)
        vitesses = torch.tensor([batch[i][1][2] for i in idx], dtype=torch.float32, device=self.device)
        actions = (pions, angles, vitesses)
        
        avantages = torch.tensor(rewards - valeurs, dtype=torch.float32, device=self.device)
        
        for i in range(self.epoch):
            _, log_proba_new, valeur_pred = self.model.forward(states, actions)
            
            loss_critic = self.loss_criterion(valeur_pred, rewards)
            
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()
            
            ratio = torch.exp(log_proba_new - log_probas)
            loss_actor = -(torch.min(ratio * avantages,torch.clamp(ratio,1-self.epsilon, 1+self.epsilon) * avantages)).sum() / self.batch_size
            
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()