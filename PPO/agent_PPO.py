from model_PPO import Model, Trainer

class Agent:
    def __init__(self, gamma=0.99, batch_size=64):
        self.n_games = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.model = Model(11, 256)
        self.trainer = Trainer(self.model, batch_size=self.batch_size)
        