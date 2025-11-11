from model_PPO import Model, Trainer
import torch
import pygame
import numpy as np

class Agent:
    def __init__(self, gamma=0.99, batch_size=64, team=0, n_pions=5):
        self.n_buts = 0
        self.n_pions = n_pions
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.memory_game = []
        self.model = Model(self.n_pions*4+2, 32, 4)
        self.trainer = Trainer(self.model, batch_size=self.batch_size)
        self.team = team
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load(filename="save"+str(self.team)+".pth")
        
    def get_state(self, game):
        state = [e.position[0]/game.largeur for e in game.objets] + [e.position[1]/game.hauteur for e in game.objets]

        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def get_action(self, game) :
        state = self.get_state(game)
        
        action, log_proba, valeur = self.model.forward(state)
        log_proba = log_proba.squeeze(0)
        valeur = valeur.squeeze(0)
        
        pion, angle, vitesse = action
        pion, angle, vitesse = pion.squeeze(0), angle.squeeze(0), vitesse.squeeze(0)
        action = (pion, angle, vitesse)
        
        self.memory_game.append([state, action, valeur, log_proba])
        # batch = (states, actions, valeurs , log_probas, rewards)
        
        pion, angle, vitesse = pion.cpu().item(), angle.cpu().item(), vitesse.cpu().item()
        
        game.objets[pion + self.team * self.n_pions].vitesse = pygame.Vector2(vitesse * np.cos(angle), vitesse * np.sin(angle))

    def fin(self, reward) :
        for e in self.memory_game :
            self.memory.append(e+[reward])
        self.memory_game = []
        if len(self.memory) > self.batch_size :
            self.trainer.train_step(self.memory)
            self.memory = []
        self.n_buts += 1 if reward == 1 else 0
        print("Team",self.team,":",self.n_buts,"buts")
        if self.n_buts%10 == 0 :
            self.save(filename="save"+str(self.team)+".pth")
    
    def save(self, filename="save.pth"):
        torch.save({
        'actor': self.model.actor.state_dict(),
        'critic': self.model.critic.state_dict(),
        "n_but": self.n_buts}, filename)
        print(f"💾 Modèle sauvegardé dans {filename}")
        
    def load(self, filename="save.pth"):
        try:
            sauvegarde = torch.load(filename)
            self.model.actor.load_state_dict(sauvegarde["actor"])
            self.model.critic.load_state_dict(sauvegarde["critic"])
            self.n_buts = sauvegarde["n_buts"]
            
            print(f"✅ model chargée depuis {filename}")
        except Exception as e:
            print("❌ Erreur lors du chargement du model :", e)
            return None