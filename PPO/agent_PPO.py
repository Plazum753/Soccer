from model_PPO import Model, Trainer
from soccer_PPO import Game
import torch

class Agent:
    def __init__(self, gamma=0.99, batch_size=64, team=0):
        self.n_games = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.model = Model(11, 256)
        self.trainer = Trainer(self.model, batch_size=self.batch_size)
        self.team = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_state(self, game):
        state = [(e.position[0], e.position[1]) for e in game.pions[self.team]]
        state += [(e.position[0], e.position[1]) for e in game.pions[self.team-1]]
        state.append(game.balle.position[0])
        state.append(game.balle.position[1])
        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def get_action(self, game) :
        state = self.get_state(game)
        
        action, log_proba, valeur = self.model.forward(state)
        
        self.memory.append([state, action, log_proba, valeur]) #TODO : reward
        # batch = (states, actions, rewards, log_probas, valeurs)
        
def train():
    game = Game()