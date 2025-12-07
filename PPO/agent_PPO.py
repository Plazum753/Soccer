from model_PPO import Model, Trainer
import torch
import numpy as np

class Agent:
    def __init__(self, gamma=0.7, lambda_=0.95, batch_size=256, team=0, n_pions=5):
        self.team = team
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_pions = n_pions
        self.n_buts = 0
        self.n_pions = n_pions
        self.batch_size = batch_size
        self.memory = []
        self.memory_game = []
        self.model = Model(self.n_pions*4+2, 256, 3, n_pions=n_pions, lr=1e-5*(10*self.team))
        self.trainer = Trainer(self.model, batch_size=self.batch_size, n_pions=n_pions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load(filename="save"+str(self.team)+".pth")
        
    def get_state(self, game):
        state = [e.position[0]/game.largeur for e in game.objets] + [e.position[1]/game.hauteur for e in game.objets]

        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def get_action(self, game) :
        state = self.get_state(game)
        
        with torch.no_grad():
            action, log_proba, valeur, _ = self.model.forward(state, self.n_pions, training=False)
        log_proba = log_proba.squeeze(0)
        valeur = valeur.squeeze(0)
        
        pion, dx, dy, v = action
        pion, dx, dy, v = pion.squeeze(0).detach().cpu(), dx.squeeze(0).detach().cpu(), dy.squeeze(0).detach().cpu(), v.squeeze(0).detach().cpu()
        action = (pion, dx, dy, v)
        state, log_proba, valeur = state.detach().cpu(), log_proba.detach().cpu(), valeur.detach().cpu()
        
        self.memory_game.append([state, action, valeur, log_proba, 0])
        
        pion, dx, dy, v = pion.item(), dx.item(), dy.item(), v.item()
        #print("prob :",torch.exp(log_proba))
        
        v = 19 / (1 + np.exp(-v)) + 1

        angle = np.arctan2(dy, dx)
        
        game.objets[pion + self.team * self.n_pions].vitesse = np.array([np.cos(angle) * v, np.sin(angle) * v], dtype=np.float64)


    def avantage(self):
        rewards = np.array([e[-1] for e in self.memory_game], dtype=np.float32)
        valeurs = np.array([e[2] for e in self.memory_game], dtype=np.float32)
        
        gae = rewards[-1] - valeurs[-1]
        returns = [gae + valeurs[-1]]
        
        for i in reversed(range(len(rewards)-1)) :
            delta = rewards[i] + self.gamma * valeurs[i+1] - valeurs[i]
            gae = delta + self.gamma * self.lambda_ * gae
            returns.insert(0, gae + valeurs[i])
        
        avantages = np.array(returns) - valeurs
        avantages = (avantages - np.mean(avantages)) / (np.std(avantages) + 1e-15)
        
        for i in range(len(self.memory_game)) :
            self.memory_game[i][-1] = returns[i]
            self.memory_game[i][2] = avantages[i]
    
    def fin(self, score) :
        if self.memory_game :
            self.avantage()
        
        for i in range(len(self.memory_game)) :
            self.memory.append(self.memory_game[i])
        self.memory_game = []
        
        if len(self.memory) > self.batch_size :
            self.trainer.train_step(self.memory)
            self.memory = []
            
        self.n_buts += 1 if score == 1 else 0
        print("Team",self.team,":",self.n_buts,"buts")
        
        if self.n_buts%10 == 0 :
            self.save(filename="save"+str(self.team)+".pth")
    
    def save(self, filename="save.pth"):
        torch.save({
        'actor': self.model.actor.state_dict(),
        'critic': self.model.critic.state_dict(),
        'optimizer': self.model.optimizer.state_dict(),
        "n_buts": self.n_buts}, filename)
        print(f"💾 Modèle {str(self.team)} sauvegardé dans {filename}")
        
    def load(self, filename="save.pth"):
        try:
            sauvegarde = torch.load(filename)
            self.model.actor.load_state_dict(sauvegarde["actor"])
            self.model.critic.load_state_dict(sauvegarde["critic"])
            self.model.optimizer.load_state_dict(sauvegarde["optimizer"])
            self.n_buts = sauvegarde["n_buts"]
            
            self.model.actor.to(self.model.device)
            self.model.critic.to(self.model.device)
            
            print(f"✅ model chargée depuis {filename}")
        except Exception as e:
            print("❌ Erreur lors du chargement du model :", e)
            return None
        