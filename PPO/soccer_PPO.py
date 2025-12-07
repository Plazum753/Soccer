import numpy as np
import pygame
from sys import exit
from numba import njit
import random
from agent_PPO import Agent
from info import plot

# import cProfile, pstats

pygame.init()

largeur = 700
hauteur = 900

terrain = pygame.display.set_mode((largeur,hauteur))
pygame.display.set_caption("soccer")
terrain_array = pygame.surfarray.array3d(terrain)

training = True

SPEED = 60
clock = pygame.time.Clock()

def texte(text, position, color = (0,0,0), taille = 30):
    police=pygame.font.SysFont('arial',taille)
    texte=police.render(text,True,color)
    zoneTexte = texte.get_rect()
    zoneTexte.center = position
    terrain.blit(texte,zoneTexte)

def affiche(largeur, hauteur, objets, score):
    terrain.fill((50,180,50))
    pygame.draw.rect(terrain,(25,220,25),(largeur*0.1,hauteur*0.1,largeur*0.8, hauteur*0.8), border_radius=30)
    pygame.draw.rect(terrain,(25,220,25),(largeur*0.35,hauteur*0.03,largeur*0.3, hauteur*0.94), border_radius=30)
    texte(str(score[0]), (largeur*0.05,hauteur*0.95))
    texte(str(score[1]), (largeur*0.05,hauteur*0.05))

    for e in objets :
        e.image()
        
def isbord(x, y, couleur=(50,180,50)):
    pixel = terrain.get_at((x, y)) # donne la couleur du pixel

    if pixel == couleur :
        return True
    return False

def point_new(point_old):
    test = ((-2,-2),(-2,0),(-2,2),(0,2),(2,2),(2,0),(2,-2),(0,-2))
    
    compteur = 0
    while isbord(point_old[0]+test[compteur][0], point_old[1]+test[compteur][1]) :
        compteur = (compteur+1)%8
    while not isbord(point_old[0]+test[compteur][0], point_old[1]+test[compteur][1]) :
        compteur = (compteur+1)%8
    return (point_old[0]+test[compteur][0], point_old[1]+test[compteur][1])
        
def centre_piste(largeur,hauteur):
    bord = []
    point = (largeur//10-1,hauteur//2) # premier point

    while point not in bord :
        bord.append(point)
        point = point_new(point)
        # pygame.draw.rect(terrain,(255,0,0),(point[0],point[1],2, 2))
        # pygame.display.update()
        # clock.tick(SPEED)
    return np.array(bord)

terrain.fill((50,180,50)) # il faut que le terrain soit affiché pour que les fonctions fonctionnent
pygame.draw.rect(terrain,(25,220,25),(largeur*0.1,hauteur*0.1,largeur*0.8, hauteur*0.8), border_radius=30)
pygame.draw.rect(terrain,(25,220,25),(largeur*0.35,hauteur*0.03,largeur*0.3, hauteur*0.94), border_radius=30)

terrain_array = pygame.surfarray.array3d(terrain)
terrain_array = np.array([[r for r,g,b in row] for row in terrain_array])
bord = centre_piste(largeur,hauteur)

texte("Chargement...",(350,400),taille=50)
pygame.display.update()

@njit
def deplacement(p, r, m, v, bord, terrain_array, largeur, hauteur):
    diff = p[:, None, :] - bord[None,: , :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    ind = np.argmin(dist, axis=1)
    
    mini = np.empty(len(p), dtype=np.float64)
    for i in range(len(p)):
        mini[i] = dist[i, ind[i]]
        
    mask = mini < r
    x = min(max(int(p[-1, 0]),0),largeur-1)
    y = min(max(int(p[-1, 1]),0),hauteur-1)
    if terrain_array[x, y] != 25 : # si balle sortie
        mask[-1] += 1
    
    ind_collision = np.where(mask)
    
    for objet, distance, b in zip(ind_collision[0], mini[ind_collision[0]], ind[ind_collision[0]]):
        # séparer les deux objet
        overlap = abs(r[objet] - distance)
        direction = p[objet] - bord[b]
        direction /= np.linalg.norm(direction) + 1e-15
        
        x = min(max(int(p[objet,0]),0),largeur-1)
        y = min(max(int(p[objet,1]),0),hauteur-1)
        
        if terrain_array[x, y] == 25 : 
            p[objet] += direction * overlap
        else : # balle dans le mur
            p[objet] -= direction * overlap
        
        # calcul le choc
        n = (p[objet] - bord[b])
        n /= np.linalg.norm(n) + 1e-15
        t = np.array([-n[1],n[0]])
        
        vo_n = np.dot(v[objet],n)
        vo_t = np.dot(v[objet], t)
        
        vo_n_new = - vo_n
        
        v[objet] = vo_n_new * n + vo_t * t
        
# =============================================================================
    
    diff = p[:, None, :] - p[None,: , :] #(N,N,2)
    dist = np.sqrt(np.sum(diff**2, axis=2)) #(N, N)
    
    mask = dist < (r[:, None] + r [None, :]) # (N, N)
    
    i_idx, j_idx = np.where(np.triu(mask, 1))
    
    for i, j in zip(i_idx, j_idx) :
        n = (p[i] - p[j])
        n /= np.linalg.norm(n)
        t = np.array([-n[1],n[0]])
        
        vi_n = np.dot(v[i], n)
        vj_n = np.dot(v[j], n)
        vi_t = np.dot(v[i], t)
        vj_t = np.dot(v[j], t)
        
        vi_n_new = (2 * m[j] * vj_n + vi_n * (m[i] - m[j])) / (m[i] + m[j])
        vj_n_new = (2 * m[i] * vi_n + vj_n * (m[j] - m[i])) / (m[i] + m[j])
        
        v[i] = vi_n_new * n + vi_t * t 
        v[j] = vj_n_new * n + vj_t * t
        
        # séparer les deux objet
        overlap = r[i] + r[j] - dist[i,j]
        direction = p[i] - p[j]
        direction /= np.linalg.norm(direction) + 1e-15

        p[i] += direction * overlap / 2
        p[j] -= direction * overlap / 2
        
    # =============================================================================
    
    p += v
    
    mask = np.sum(v**2, axis=1) < 0.05    
    v[mask] = 0
    v *= 0.985
    
    return p, v

print("chargement...")

p = np.zeros((2,2), dtype=np.float64)
r = np.zeros((2), dtype=np.float64)
m = np.zeros((2), dtype=np.float64)
v = np.zeros((2,2), dtype=np.float64)
deplacement(p, r, m, v, bord, terrain_array,1,1)
    

class Pion :
    def __init__(self, team, position):
        self.position = np.array([position[0], position[1]], dtype=np.float64)
        self.vitesse = np.array([0, 0], dtype=np.float64)
        self.poid = 1.0
        self.rayon = 25.0
        self.team = team
        self.couleur = np.array([200,0,0]) if team == 0 else np.array([0,0,200])
    
    def image(self):
        pygame.draw.circle(terrain, self.couleur, self.position.astype(np.int16), self.rayon)
        pygame.draw.circle(terrain, self.couleur*1.2, self.position.astype(np.int16), self.rayon*0.8, width= 7)

    def actif(self, objets, score):
        pygame.draw.circle(terrain, (255,255,255), self.position.astype(np.int16), 32, 5)
        pygame.display.update()

        souris = None

        while souris == None :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit() 
            if pygame.mouse.get_pressed()[0] :
                souris = pygame.Vector2(pygame.mouse.get_pos())
        
        while pygame.mouse.get_pressed()[0] :
            pygame.event.pump()
            
            vecteur =  pygame.Vector2(souris[0] - pygame.mouse.get_pos()[0], souris[1] - pygame.mouse.get_pos()[1])
            vecteur.scale_to_length(200) if vecteur.length_squared() > 200**2 else None

            affiche(largeur,hauteur, objets, score)
            pygame.draw.circle(terrain, (255,255,255), self.position, 32, 5)
            # tracer de la ligne
            pygame.draw.line(terrain, (255,255,0), souris, souris - vecteur, width=7)
            pygame.draw.circle(terrain, (255,255,0), souris, 3.5)
            pygame.draw.circle(terrain, (255,255,0), souris - vecteur, 3.5)
            pygame.display.update()
            
        vecteur /= 10
        if vecteur.length_squared() < 1 :
            affiche(largeur,hauteur, objets, score)
            pygame.display.update()
            return 0
        self.vitesse = vecteur
        return 1
    
    
class Balle :
    def __init__(self, largeur, hauteur):
        self.position = np.array([largeur/2,hauteur/2], dtype=np.float64)
        self.vitesse = np.array([0, 0], dtype=np.float64)
        self.poid = 0.5
        self.rayon = 100#15.0

    
    def image(self):
        pygame.draw.circle(terrain, (255,255,255), self.position.astype(np.int16), self.rayon)

    def but(self, largeur, hauteur):
        if largeur*0.35 < self.position[0] < largeur*0.65 :
            if self.position[1] < hauteur*0.2 :#0.1
                return 1
            elif self.position[1] > hauteur*0.8 :#0.9
                return 0
        return None
  
    
class Game :
    def __init__(self,largeur=700, hauteur=900, joueurs = ("ia","ia"), training=True, agents=None):
        self.joueurs = joueurs
        
        self.agents = [Agent(team=0, n_pions=1), Agent(team=1, n_pions=1)] if agents == None else agents
        self.training = training
        
        self.score = [0, 0]
        
        self.largeur = largeur
        self.hauteur = hauteur
        
        self.reset()

        affiche(self.largeur,self.hauteur, self.objets, self.score)
        pygame.display.update()
        
        self.tour = random.randint(0, 1)
        
        affiche(self.largeur, self.hauteur, self.objets, self.score)
        pygame.display.update()
    
    def reset(self) :
        self.pions = []
        
        pions_0 = []
        pions_0.append(Pion(0,(self.largeur*0.5,self.hauteur*0.3)))
        # for i in range(1,3):
        #     pions_0.append(Pion(0,(self.largeur*(1/3)*i,self.hauteur*0.2)))
        # for i in range(2,4):
        #     pions_0.append(Pion(0,(self.largeur*0.2*i,self.hauteur*0.1)))
        
        pions_1 = []
        pions_1.append(Pion(1,(self.largeur*0.5,self.hauteur*0.7)))
        # for i in range(1,3):
        #     pions_1.append(Pion(1,(self.largeur*(1/3)*i,self.hauteur*0.8)))
        # for i in range(2,4):
        #     pions_1.append(Pion(1,(self.largeur*0.2*i,self.hauteur*0.9)))
            
        self.balle = Balle(self.largeur,self.hauteur)
        
        self.pions = [pions_0, pions_1]
        self.objets = pions_0+pions_1+[self.balle]
        
        self.n_tir = 0
        self.n_touches = 0
        
    def coup(self):
        while True :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            if pygame.mouse.get_pressed()[0] :
                while pygame.mouse.get_pressed()[0] :
                    pygame.event.pump()
                souris = pygame.mouse.get_pos()
                for e in self.pions[self.tour] :
                    if (e.position[0] - souris[0])**2 + (e.position[1] - souris[1])**2 <= 625 : # 25**2 = 625
                        self.tour = (self.tour+e.actif(self.objets, self.score))%2  
                        return
                    
    def partie(self, terrain_array, bord):  
        while self.score[0] < 3 and self.score[1] < 3 :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            if training == False :
                affiche(self.largeur, self.hauteur, self.objets, self.score)
                pygame.display.update()
                    
            if self.joueurs[self.tour] == "ia":
                self.agents[self.tour].get_action(self) 
                self.tour = (self.tour+1)%2
                if training == False :
                    clock.tick(1)
            else :
                self.coup()
                
            self.n_tir += 1
            
            balle_old = self.objets[-1].position[1]
            
            # dist_old = self.objets[-1].position - self.objets[(self.tour+1)%2].position
            # dist_old = np.dot(dist_old, dist_old)
            
            # affiche(self.largeur, self.hauteur, self.objets, self.score)
            # pygame.display.update()

            while any(np.linalg.norm(o.vitesse) != 0 for o in self.objets):
                pygame.event.pump()

                p = np.array([o.position for o in self.objets], dtype=np.float64)
                r = np.array([o.rayon for o in self.objets], dtype=np.float64)
                m = np.array([o.poid for o in self.objets], dtype=np.float64)
                v = np.array([o.vitesse for o in self.objets], dtype=np.float64)
                
                p, v = deplacement(p, r, m, v, bord, terrain_array, self.largeur, self.hauteur)
                    
                for i in range(len(self.objets)) :
                    self.objets[i].position = p[i]
                    self.objets[i].vitesse = v[i]
                
                
                but_val = self.objets[-1].but(largeur, hauteur)
                if but_val != None :
                    
                    self.score[(but_val+1)%2] += 1
                    if training == True :
                        n_tir_plot.append(self.n_tir)
                        n_touches_plot.append(self.n_touches)
                        if self.agents[but_val].memory_game :
                            self.agents[but_val].memory_game[-1][-1] += 1 # reward
                        if self.agents[but_val-1].memory_game :
                            self.agents[but_val-1].memory_game[-1][-1] -= 1 # reward
                        self.agents[but_val].fin(1)
                        self.agents[but_val-1].fin(-1)
                        print()
                    
                    self.reset()
                    self.tour = (but_val+1)%2
                    
                if training == False : 
                    affiche(self.largeur, self.hauteur, self.objets, self.score)
                    pygame.display.update()
                    clock.tick(SPEED)
            
            
            if training == True : 
                if balle_old != self.objets[-1].position[1] :
                    if self.agents[self.tour-1].memory_game :
                        self.agents[self.tour-1].memory_game[-1][-1] += 1 
                    self.n_touches += 1
                    
                # dist = self.objets[-1].position - self.objets[(self.tour+1)%2].position
                # dist = np.dot(dist,dist)
                # if dist < dist_old :
                #     if self.agents[self.tour-1].memory_game :
                #         self.agents[self.tour-1].memory_game[-1][-1] += 0.3
                # else :
                #     if self.agents[self.tour-1].memory_game :
                #         self.agents[self.tour-1].memory_game[-1][-1] -= 0.3
                
# profiler = cProfile.Profile()
# profiler.enable()

if training == True :
    n_tir_plot = []
    n_touches_plot = []
    agents = [Agent(team=0, n_pions=1), Agent(team=1, n_pions=1)]
    while True :
        game = Game(agents=agents)
        game.partie(terrain_array, bord)
        plot(n_tir_plot, n_touches_plot)
        
        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(50)
        
        
game = Game(joueurs = ("ia","hu"), training=False)
game.partie(terrain_array, bord)

pygame.quit()
exit()
    
