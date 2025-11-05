import numpy as np
import pygame

pygame.init()

largeur = 700
hauteur = 900

terrain = pygame.display.set_mode((largeur,hauteur))
pygame.display.set_caption("soccer")

frottement = 0.5

def affiche_terrain(largeur,hauteur):
    terrain.fill((50,180,50))
    pygame.draw.rect(terrain,(25,220,25),(largeur/10,hauteur/10,largeur-largeur/5, hauteur-hauteur/5), border_radius=30)

class Pion :
    def __init__(self, team, position):
        self.position = pygame.Vector2(position[0], position[1])
        self.vitesse = pygame.Vector2(0, 0)
        self.poid = 10
    
    def image(self):
        pygame.draw.circle(terrain, (255,0,0), self.position, 25)
        
    def actif(self):
        print("salut")
    
    def deplacement(self):
        pass
    def choc(self):
        pass
    
class Balle :
    def __init__(self, largeur, hauteur):
        self.position = pygame.Vector2(largeur/2,hauteur/2)
        self.vitesse = pygame.Vector2(0, 0)
        self.poid = 5

    def but(self):
        pass
    
    def deplacement(self):
        pass
    def choc(self):
        pass

score = (0, 0)

affiche_terrain(largeur,hauteur)

pions = []
for i in range(2):
    pions.append(Pion(0,(100*(i+1),200*(i+1))))
    pions[i].image()

while True:
    if pygame.event.get(pygame.QUIT) :
        pygame.quit()
        exit()
    pygame.display.update()
    
    if pygame.event == pygame.MOUSEBUTTONDOWN :
        souris = pygame.mouse.get_pos()
        print(souris)
        for e in pions :
            print(e.position.x, e.position.y)
            if (e.position.x - souris[0])**2 + (e.position.y - souris[1])**2 <= 25 :
                e.actif()
                break
    
    
