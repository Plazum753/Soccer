import numpy as np
import pygame
from sys import exit
from numba import njit
import random
from agent_PPO import Agent

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
bord = centre_piste(largeur,hauteur)

@njit
def choc_mur(va, a, ma, r, bord, terrain_array) :
    # trouve le point le plus proche
    mini = np.linalg.norm(bord[0] - a)
    ind = 0
    for i in range(1,len(bord)) :
        if np.linalg.norm(bord[i] - a) < mini :
            mini = np.linalg.norm(bord[i] - a)
            ind = i
    b = bord[ind]
    
    if mini > r :
        return a, va
    
    # séparer les deux objet
    overlap = r - mini
    direction = a - b
    direction /= np.linalg.norm(direction)
    
    if np.all(terrain_array[int(a[0]), int(a[1])] == np.array([25,220,25])) : 
        a += direction * overlap
    else : # balle dans le mur
        a -= direction * overlap
    
    # calcul le choc
    n = (a - b)
    n /= np.linalg.norm(n)
    t = np.array([-n[1],n[0]])
    
    va_n = np.dot(va,n)
    va_t = np.dot(va, t)
    
    va_n_new = - va_n
    
    va_new = va_n_new * n + va_t * t
    
    return a, va_new

choc_mur(np.array([1.0,1.0]),np.array([1.0,1.0]),1.0,1.0,bord,terrain_array)
    
@njit
def calcul_choc(va, vb, a, b, ma, mb):
    n = (a - b)
    n /= np.linalg.norm(n)
    t = np.array([-n[1],n[0]])
    
    va_n = np.dot(va, n)
    vb_n = np.dot(vb, n)
    va_t = np.dot(va, t)
    vb_t = np.dot(vb, t)
    
    va_n_new = (2 * mb * vb_n + va_n * (ma - mb)) / (ma + mb)
    vb_n_new = (2 * ma * va_n + vb_n * (mb - ma)) / (ma + mb)
    
    va = va_n_new * n + va_t * t 
    vb = vb_n_new * n + vb_t * t
    
    return va, vb

calcul_choc(np.array([1.0,1.0]),np.array([1.0,1.0]),np.array([1.0,1.0]),np.array([1.0,1.0]),1.0,1.0)
    

class Pion :
    def __init__(self, team, position):
        self.position = pygame.Vector2(position[0], position[1])
        self.vitesse = pygame.Vector2(0, 0)
        self.poid = 0.5
        self.rayon = 25.0
        self.tap = False
        self.team = team
        self.couleur = np.array([200,0,0]) if team == 0 else np.array([0,0,200])
    
    def image(self):
        pygame.draw.circle(terrain, self.couleur, self.position, self.rayon)
        pygame.draw.circle(terrain, self.couleur*1.2, self.position, self.rayon*0.8, width= 7)

    def actif(self, objets, score):
        pygame.draw.circle(terrain, (255,255,255), self.position, 32, 5)
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
    
    def deplacement(self, frottement):
        self.position += self.vitesse
        self.vitesse *= frottement if self.vitesse.length_squared() > 0.05 else 0
    
    def choc(self, objets, bord, terrain_array):
        a, va = choc_mur(np.array([self.vitesse.x, self.vitesse.y]), np.array([self.position.x, self.position.y]), self.poid, self.rayon, bord, terrain_array)
        self.vitesse = pygame.Vector2(va[0], va[1])
        self.position = pygame.Vector2(a[0],a[1])
        
        i = 0
        for e in objets :
            if e.rayon + self.rayon > (e.position - self.position).length() and e is not self and self.tap == False :
                # séparer les deux objet
                overlap = e.rayon + self.rayon - (e.position - self.position).length()
                direction = (self.position - e.position).normalize()
                self.position += direction * overlap / 2
                e.position -= direction * overlap / 2
                
                # calcul force du choc
                va, vb = calcul_choc(np.array([self.vitesse.x, self.vitesse.y]), np.array([e.vitesse.x, e.vitesse.y]), np.array([self.position.x, self.position.y]), np.array([e.position.x, e.position.y]), self.poid, e.poid)
                self.vitesse = pygame.Vector2(va[0], va[1])
                e.vitesse = pygame.Vector2(vb[0], vb[1])
                e.tap = True
            else :
                i += 1
        if i == len(objets) :
            self.tap = False
    
class Balle :
    def __init__(self, largeur, hauteur):
        self.position = pygame.Vector2(largeur/2,hauteur/2)
        self.vitesse = pygame.Vector2(0, 0)
        self.poid = 0.25
        self.rayon = 15.0
        self.tap = False

    
    def image(self):
        pygame.draw.circle(terrain, (255,255,255), self.position, self.rayon)

    def but(self, largeur, hauteur):
        if largeur*0.35 < self.position.x < largeur*65 :
            if self.position.y < hauteur*0.1 :
                return 0
            elif self.position.y > hauteur*0.9 :
                return 1
        return None
    
    def deplacement(self, frottement):
        self.position += self.vitesse
        self.vitesse *= frottement if self.vitesse.length_squared() > 0.05 else 0
        
    def choc(self, objets, bord, terrain_array):
        a, va = choc_mur(np.array([self.vitesse.x, self.vitesse.y]), np.array([self.position.x, self.position.y]), self.poid, self.rayon, bord, terrain_array)
        self.vitesse = pygame.Vector2(va[0], va[1])
        self.position = pygame.Vector2(a[0],a[1])

        i = 0
        for e in objets :
            if e.rayon + self.rayon > (e.position - self.position).length() and e is not self and self.tap == False :
                # séparer les deux objet
                overlap = e.rayon + self.rayon - (e.position - self.position).length()
                direction = (self.position - e.position).normalize()
                self.position += direction * overlap / 2
                e.position -= direction * overlap / 2
                
                # calcul force du choc
                va, vb = calcul_choc(np.array([self.vitesse.x, self.vitesse.y]), np.array([e.vitesse.x, e.vitesse.y]), np.array([self.position.x, self.position.y]), np.array([e.position.x, e.position.y]), self.poid, e.poid)
                self.vitesse = pygame.Vector2(va[0], va[1])
                e.vitesse = pygame.Vector2(vb[0], vb[1])
                e.tap = True
            else :
                i += 1
        if i == len(objets) :
            self.tap = False
  
class Game :
    def __init__(self,largeur=700, hauteur=900, joueurs = ("ia","ia"), training=True):
        self.frottement = 0.985
        self.joueurs = joueurs
        
        self.agents = [Agent(team=0), Agent(team=1)]
        self.training = training
        
        self.score = [0, 0]
        
        self.largeur = largeur
        self.hauteur = hauteur
        
        self.reset()

        affiche(self.largeur,self.hauteur, self.objets, self.score)
        pygame.display.update()
        
        self.tour = random.randint(0, 1)
    
    def reset(self) :
        self.pions = []
        
        pions_0 = []
        pions_0.append(Pion(0,(self.largeur*0.5,self.hauteur*0.3)))
        for i in range(1,3):
            pions_0.append(Pion(0,(self.largeur*(1/3)*i,self.hauteur*0.2)))
        for i in range(2,4):
            pions_0.append(Pion(0,(self.largeur*0.2*i,self.hauteur*0.1)))
        
        pions_1 = []
        pions_1.append(Pion(1,(self.largeur*0.5,self.hauteur*0.7)))
        for i in range(1,3):
            pions_1.append(Pion(1,(self.largeur*(1/3)*i,self.hauteur*0.8)))
        for i in range(2,4):
            pions_1.append(Pion(1,(self.largeur*0.2*i,self.hauteur*0.9)))
            
        self.balle = Balle(self.largeur,self.hauteur)
        
        self.pions = [pions_0, pions_1]
        self.objets = pions_0+pions_1+[self.balle]
        
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
                    if (e.position.x - souris[0])**2 + (e.position.y - souris[1])**2 <= 625 : # 25**2 = 625
                        self.tour = (self.tour+e.actif(self.objets, self.score))%2  
                        return
                    
    def partie(self, terrain_array, bord):  
        while self.score[0] < 3 and self.score[1] < 3 :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
            affiche(self.largeur, self.hauteur, self.objets, self.score)
            pygame.display.update()
                    
            if self.joueurs[self.tour] == "ia":
                self.agents[self.tour].get_action(self) 
                self.tour = (self.tour+1)%2
                if training == False :
                    clock.tick(1)
            else :
                self.coup()
                
            while not np.all([_.vitesse == (0,0) for _ in self.objets]) :
                pygame.event.pump()
                for e in self.objets :
                    e.deplacement(self.frottement)
                    e.choc(self.objets, bord, terrain_array)
                    
                    but_val = self.objets[-1].but(largeur, hauteur)
                    if but_val != None :
                        self.score[but_val] += 1
                        self.agents[but_val].fin(1)
                        self.agents[but_val-1].fin(-1)
                        self.reset()
                        self.tour = but_val
                    if training == False :
                        affiche(self.largeur, self.hauteur, self.objets, self.score)
                        pygame.display.update()
                    
                if training == False :
                    clock.tick(SPEED)

if training == True :
    while True :
        game = Game()
        game.partie(terrain_array, bord)
        
game = Game(joueurs = ("ia","ia"), training=False)
game.partie(terrain_array, bord)

pygame.quit()
exit()
    
