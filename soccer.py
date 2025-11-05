import numpy as np
import pygame
from sys import exit
from numba import njit

pygame.init()

largeur = 700
hauteur = 900

terrain = pygame.display.set_mode((largeur,hauteur))
pygame.display.set_caption("soccer")

frottement = 0.95
SPEED = 120
clock = pygame.time.Clock()
score = (0, 0)



def affiche(largeur, hauteur, objets):
    terrain.fill((50,180,50))
    pygame.draw.rect(terrain,(25,220,25),(largeur/10,hauteur/10,largeur-largeur/5, hauteur-hauteur/5), border_radius=30)
    
    for e in objets :
        e.image()

@njit
def calcul_choc(va, vb, a, b, ma, mb):
    va = np.array(va)
    vb = np.array(vb)
    a = np.array(a)
    b = np.array(b)
    
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
    

class Pion :
    def __init__(self, team, position):
        self.position = pygame.Vector2(position[0], position[1])
        self.vitesse = pygame.Vector2(0, 0)
        self.poid = 0.5
        self.rayon = 25
        self.tap = False
    
    def image(self):
        pygame.draw.circle(terrain, (255,0,0), self.position, self.rayon)
        
    def actif(self, objets):
        pygame.draw.circle(terrain, (255,255,255), self.position, 32, 5)
        pygame.display.update()

        souris = None

        while souris == None :
            pygame.event.pump()
            if pygame.mouse.get_pressed()[0] :
                souris = pygame.mouse.get_pos()
        
        while pygame.mouse.get_pressed()[0] :
            pygame.event.pump()
            affiche(largeur,hauteur, objets)
            pygame.draw.line(terrain, (255,255,0), souris, pygame.mouse.get_pos(), width=3)
            pygame.display.update() #TODO ajouter un max de vitesse initiale


            
        coup = (souris[0] - pygame.mouse.get_pos()[0], souris[1] - pygame.mouse.get_pos()[1])
                
        self.vitesse.x, self.vitesse.y = coup[0]/10, coup[1]/10
    
    def deplacement(self, frottement):
        self.position += self.vitesse
        self.vitesse *= frottement if self.vitesse.length_squared() > 0.05 else 0
    
    def choc(self, objets):
        i = 0
        for e in objets :
            if e.rayon + self.rayon > (e.position - self.position).length() and e is not self and self.tap == False :
                # séparer les deux objet
                overlap = e.rayon + self.rayon - (e.position - self.position).length()
                direction = (self.position - e.position).normalize()
                self.position += direction * overlap / 2
                e.position -= direction * overlap / 2
                
                # calcul force du choc
                va, vb = calcul_choc((self.vitesse.x, self.vitesse.y), (e.vitesse.x, e.vitesse.y), (self.position.x, self.position.y), (e.position.x, e.position.y), self.poid, e.poid)
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
        self.rayon = 15
    
    def image(self):
        pygame.draw.circle(terrain, (255,255,255), self.position, self.rayon)

    def but(self):
        pass
    
    def deplacement(self):
        pass
    def choc(self):
        pass


pions = []
for i in range(3):
    pions.append(Pion(0,(100*(i+1),200*(i+1))))
    
balle = Balle(largeur, hauteur)
objets = pions+[balle]

affiche(largeur,hauteur, objets)
pygame.display.update()

while True:
    pygame.event.pump()
    if pygame.event.get(pygame.QUIT) :
        pygame.quit()
        exit()
        

    if pygame.mouse.get_pressed()[0] :
        while pygame.mouse.get_pressed()[0] :
            pygame.event.pump()
        souris = pygame.mouse.get_pos()
        for e in pions :
            if (e.position.x - souris[0])**2 + (e.position.y - souris[1])**2 <= 625 : # 25**2 = 625
                e.actif(objets)
                
                while not np.all([_.vitesse == (0,0) for _ in pions]) :
                    for e in pions :
                        e.deplacement(frottement)
                        e.choc(pions)
                        balle.choc()
                        
                        affiche(largeur,hauteur, objets)
                        pygame.display.update()
                        
                        clock.tick(SPEED)
                break

    
    
    
    
