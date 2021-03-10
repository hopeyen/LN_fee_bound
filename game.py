# template of PygameGame sourced from Lucas Peraza
# http://blog.lukasperaza.com/getting-started-with-pygame/
# Recursion Permutation sourced from 112 website
# https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html#permutations
# Minimax Algorithm structure from 112 lecture slides
#https://docs.google.com/presentation/d/1kqeJd4w05TiWf3-h2dBMfZvwjlPisZnMnyPJ7ymfcIo/edit#slide=id.g368c43f61e_0_273E

from helper import *
import pygame
import os
from Tkinter import *
os.environ["SDL_VIDEODRIVER"] = "dummy"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHTBLUE = (204, 229, 255)

class PygameGame(object):
    def init(self):
        self.start = True
        self.playerPage = False
        self.play = False
        self.playMode1 = False
        self.playMode2 = False
        self.end = False
        self.player = True # alternate between 2 players
        self.win1 = False
        self.win2 = False
        self.curLine = []
        self.level = ""
        self.p1Name = ""
        self.p2Name = ""
        self.p1Color = RED
        self.p2Color = BLACK
        self.inputCount = 0
        self.lastLine = []
        self.sol1 = []
        self.sol2 = []
        self.highest1 =  False
        self.highest2 = False
        self.instruction = pygame.image.load("Instruction.png")
        self.helpTimer = 300 # every 5 seconds
        self.timer = 0
        self.showHelp = False
        self.giveHelp = False
        self.allowRemove = False
        self.hint = []
    
    def mousePressed(self, x, y):
        if self.start:
           self.start = False
           self.playerPage = True
        
        if self.play and not (self.win1 or self.win2):
            self.curLine = [(x,y), (x,y)]

    def mouseReleased(self, x, y):
        if self.play and not (self.win1 or self.win2):
            self.curLine[1] = pygame.mouse.get_pos()
            if self.player:
                newLine = Line(self.curLine[0][0], \
                    self.curLine[0][1], self.curLine[1][0], \
                    self.curLine[1][1], self.p1Color)
                if checkHit(newLine) and checkNew(newLine):
                    self.p1Lines.add(newLine)
                    self.lastLine = newLine
                    self.allowRemove = True
                    if self.playMode1 and \
                        len(self.p1Lines) > self.checks[0]: 
                        self.sol1 = existComplete(self.p1Lines, self.checks[0])
                        self.sol1 = allTri(self.p1Lines)
                        check = self.checks[0]
                        if check == 2:
                            if len(self.sol1) > 0: self.win1 = True
                            else: self.win1 = False
                        else:
                            self.highest1 = checkHigher(self.sol1, self.p1Lines)
                            if self.highest1 == check: self.win1 = True
                            else: self.win1 = False
                        
                        self.end = True if self.win1 else False
                        
                    if self.playMode2:
                        self.sol1 = allTri(self.p1Lines)
                        self.highest1 = checkHigher(self.sol1, self.p1Lines)
                    self.player = not self.player
                    self.timer = 0
                    
            else:
                newLine = Line(self.curLine[0][0], \
                    self.curLine[0][1], self.curLine[1][0], \
                    self.curLine[1][1], self.p2Color)
                if checkHit(newLine) and checkNew(newLine):
                    self.p2Lines.add(newLine)
                    self.lastLine = newLine
                    self.allowRemove = True
                    if self.playMode1:
                        self.sol2 = existComplete(self.p2Lines, self.checks[1])
                        self.sol2 = allTri(self.p2Lines)
                        check = self.checks[1]
                        if check == 2:
                            if len(self.sol2) > 0: self.win2 = True
                            else: self.win2 = False
                        else:
                            self.highest2 = checkHigher(self.sol2, self.p2Lines)
                            if self.highest2 == check: self.win2 = True
                            else: self.win2 = False
                       
                    if self.playMode2:
                        self.sol2 = allTri(self.p2Lines)
                        self.highest2 = checkHigher(self.sol2, self.p2Lines)
                    self.player = not self.player
                    self.timer = 0
                    
                    
            self.curLine = []
            
            if self.playMode2:
                if finished(self.p1Lines, self.p2Lines):
                    self.end = True
                    if self.highest1 > self.highest2:
                        self.win1 = True
                    elif self.highest2 > self.highest1:
                        self.win2 = True
                    else:
                        if len(self.sol1) > len(self.sol2):
                            self.win1 = True
                        elif len(self.sol2) > len(self.sol1):
                            self.win2 = True
        

    
                        
    def mouseMotion(self, x, y):
        pass

    def mouseDrag(self, x, y):
        if self.play and not (self.win1 or self.win2):
            self.curLine[1] = pygame.mouse.get_pos()
        
    def keyPressed(self, keyCode, modifier):
        if self.playerPage:
            if keyCode == 13:
                if self.inputCount <= 3:
                    if (self.inputCount == 0 and self.p1Name != "") or\
                        (self.inputCount == 1 and self.p2Name != "") or\
                        (self.inputCount == 2 and self.level != "") or\
                        (self.inputCount == 3 and (self.playMode1 or self.playMode2)):
                            self.inputCount += 1
                else:
                    self.playerPage = False
                    self.play = True
                    Level(self.level, self.dots)
                    self.checks = checkLv(int(self.level))
                    self.dots = Dot.dots
                    
                    if len(self.p1Name) >= 10:
                        if 0 <= int(self.p1Name[-9:]) <= 255255255:
                            self.p1Color = (int(self.p1Name[-9:-6]),\
                            int(self.p1Name[-6:-3]), int(self.p1Name[-3:]))
                        self.p1Name = self.p1Name[:-9]
                    if len(self.p2Name) >= 10:
                        if 0 <= int(self.p2Name[-9:]) <= 255255255:
                            self.p2Color = (int(self.p2Name[-9:-6]),\
                            int(self.p2Name[-6:-3]), int(self.p2Name[-3:]))
                        self.p2Name = self.p2Name[:-9]
                    
                    if len(self.p1Name) > 8:
                        self.p1Name = self.p1Name[:8]
                    if len(self.p2Name) > 8:
                        self.p2Name = self.p2Name[:8]
                        
                    Player(self.p1Name, self.p1Color)
                    Player(self.p2Name, self.p2Color)
                    
            else:
                if self.inputCount == 0:
                    if keyCode == 127 or keyCode == 8:
                        self.p1Name = self.p1Name[0:-1]
                    else:
                        self.p1Name += chr(keyCode)
                if self.inputCount == 1:
                    if keyCode == 127 or keyCode == 8:
                        self.p2Name = self.p2Name[0:-1]
                    else:
                        self.p2Name += chr(keyCode)
                if self.inputCount == 2:
                    if 49 <= keyCode <= 51:
                        self.level = chr(keyCode)
                if self.inputCount == 3:
                    if chr(keyCode) == "a":
                        self.playMode1 = True
                        self.playMode2 = False
                    if chr(keyCode) == "b":
                        self.playMode2 = True
                        self.playMode1 = False
            
        if self.play:
            if self.allowRemove and chr(keyCode) == "d":
                self.player = not self.player
                if self.player:
                    self.p1Lines.remove(self.lastLine)
                else: self.p2Lines.remove(self.lastLine)
                self.allowRemove = False
                
            
            if chr(keyCode) == "h":
                self.giveHelp = True

    def keyReleased(self, keyCode, modifier):
        if chr(keyCode) == "h":
            self.giveHelp = False

    def timerFired(self, dt):
        if self.play and not self.end:
            self.timer += 1
            if self.timer // self.helpTimer > 0:
                if self.playMode1:
                    if self.player:
                        self.hint = getHintMode1(self.p1Lines, self.p2Lines)
                    else:
                        self.hint = getHintMode1(self.p2Lines, self.p1Lines)
                else:
                    if self.player:
                        self.hint = getHintMode2(self.p1Lines, self.p2Lines)
                    else:
                        self.hint = getHintMode2(self.p2Lines, self.p1Lines)
                self.showHelp = True
            else:
                self.showHelp = False
            
            

    def redrawAll(self, screen):
        if self.start:
            drawStart(screen, self.font, self.instruction)
        
        if self.playerPage:
            if self.inputCount <= 3:
                drawPlayerPage(screen, self.font)
                drawSpot(screen, self.inputCount, self.font, \
                    self.p1Name, self.p2Name)
            if self.inputCount == 4:
                drawReady(screen, self.font)
            drawPlayerNames(screen, self.font, self.p1Name, \
                self.p2Name, str(self.level), self.playMode1, self.playMode2)
            
        if self.play:
            drawTitle(screen, self.font, self.playMode1, self.level)
            drawPlayerStats(screen, self.font, self.player)
            for dot in self.dots:
                dot.draw(screen)
            if len(self.curLine) > 0:
                pygame.draw.line(screen, BLACK, self.curLine[0], self.curLine[1])
            for line in self.p1Lines:
                line.draw(screen)
                #if self.playMode2:
                for sol in self.sol1:
                    if ((line.x0, line.y0)) in sol and \
                        ((line.x1, line.y1)) in sol:
                        line.drawBold(screen)
            for line in self.p2Lines:
                line.draw(screen)
                # if self.playMode2:
                for sol in self.sol2:
                    if ((line.x0, line.y0)) in sol and \
                        ((line.x1, line.y1)) in sol:
                        line.drawBold(screen)
            
            if self.playMode2:
                drawTriNum(screen, self.font, self.p1Name, \
                    self.p2Name, len(self.sol1), len(self.sol2))
            
            if self.showHelp:
                drawHelpButton(screen, self.font)
                if self.giveHelp:
                    drawHelp(screen, self.hint)
            
            
        if self.end:
            if self.playMode1:
                if self.win1: text = Player.players[1].name + " won! "
                else: text = Player.players[0].name + " won! "
                drawWinMode1(screen, self.font, text)
            if self.playMode2:
                self.sol1 = allTri(self.p1Lines)
                self.highest1 = checkHigher(self.sol1, self.p1Lines)
                if self.highest1 != 2:
                    player1Text = Player.players[0].name + \
                        " has k_" + str(self.highest1 + 1) + ". "
                else: player1Text = Player.players[0].name + \
                        " has " + str(len(self.sol1)) + " triangles. "
                self.sol2 = allTri(self.p2Lines)
                self.highest2 = checkHigher(self.sol2, self.p2Lines)
                if self.highest2!= 2:
                    player2Text = Player.players[1].name + \
                        " has k_" + str(self.highest2 + 1) + ". "
                else: player2Text = Player.players[1].name + \
                        " has " + str(len(self.sol2)) + " triangles. "
                
                if self.highest1 > self.highest2:
                    text = Player.players[0].name + " won. "
                elif self.highest2 > self.highest1:
                    text = Player.players[1].name + " won. "
                else:
                    if len(self.sol1) > len(self.sol2):
                        text = Player.players[0].name + " won. "
                    else:
                        text = Player.players[1].name + " won. "
                drawWinMode2(screen, self.font, text, player1Text, player2Text)
                    
    def isKeyPressed(self, key):
        ''' return whether a specific key is being held '''
        return self._keys.get(key, False)

    def __init__(self, width=600, height=400, fps=60, title="Ramsey's Graph"):
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self.bgColor = LIGHTBLUE
        pygame.init()

    def run(self):
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.width, self.height))
        # set the title of the window
        pygame.display.set_caption(self.title)

        self.dots = pygame.sprite.Group()
        self.p1Lines = pygame.sprite.Group()
        self.p2Lines = pygame.sprite.Group()
        self.allSpritesList = pygame.sprite.Group()
        self.hitlist = pygame.sprite.Group()
        # stores all the keys currently being held down
        self._keys = dict()

        # call game-specific initialization
        self.init()
        self.font = pygame.font.Font(None, 36)
        
        playing = True
        while playing:
            time = clock.tick(self.fps)
            self.timerFired(time)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.mousePressed(*(event.pos))
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.mouseReleased(*(event.pos))
                elif (event.type == pygame.MOUSEMOTION and
                      event.buttons == (0, 0, 0)):
                    self.mouseMotion(*(event.pos))
                elif (event.type == pygame.MOUSEMOTION and
                      event.buttons[0] == 1):
                    self.mouseDrag(*(event.pos))
                elif event.type == pygame.KEYDOWN:
                    self._keys[event.key] = True
                    self.keyPressed(event.key, event.mod)
                elif event.type == pygame.KEYUP:
                    self._keys[event.key] = False
                    self.keyReleased(event.key, event.mod)
                elif event.type == pygame.QUIT:
                    playing = False
            screen.fill(self.bgColor)
            self.redrawAll(screen)
            pygame.display.flip()
        pygame.quit()


def main():
    game = PygameGame()
    game.run()

if __name__ == '__main__':
    main()