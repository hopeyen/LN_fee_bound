import pygame
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
HINTYELLOW = (255, 255, 153)

class Dot(pygame.sprite.Sprite):
    dots = []
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([10, 10])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.x, self.y = x, y
        self.rect.x, self.rect.y = x, y
        Dot.dots.append(self)

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (self.rect.x, self.rect.y), 6)

class Line(pygame.sprite.Sprite):
    linesDrawn = set()
    def __init__(self, x0, y0, x1, y1, color):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.color = color
    
    def draw(self, screen):
        pygame.draw.line(screen, self.color, [self.x0, self.y0], \
            [self.x1, self.y1], 2)
    
    def drawBold(self, screen):
        pygame.draw.line(screen, self.color, [self.x0, self.y0], \
            [self.x1, self.y1], 4)
    
    def __repr__(self):
        return ("Line at (%s, %s) and (%s, %s)" %(str(self.x0), \
            str(self.y0), str(self.x1), str(self.y1)))
    
    # def __eq__(self, other):
    #     return isinstance(other, Line) and \
    #         ((self.x0 == other.x0 and self.x1 == other.x1 and self.y0 == other.y0 and self.y1 == other.y1) or
    #         (self.x1 == other.x1 and self.x0 == other.x1 and self.y1 == other.y0 and self.y0 == other.y1))

class Player(pygame.sprite.Sprite):
    players = []
    def __init__(self, name, color):
        super().__init__()
        self.name = name
        self.color = color
        Player.players.append(self)
        
class Level(pygame.sprite.Sprite):
    level_dict = {"1": [(250, 185), (500, 185), (375, 125), \
                        (250, 300), (375, 350), (500, 300)], 
    "2": [(405, 90), (500, 125), (540, 211), (510, 290), \
        (470, 335), (350, 335), (280, 290), (260, 210), (320, 125)],
    "3": [(360, 95), (410, 95), (455, 110), (490, 140), (515, 185), \
        (525, 230), (520, 280), (495, 320), (460, 355), (410, 370), \
        (365, 370), (320, 355), (280, 325), (255, 280), (245, 235), \
        (260, 180), (280, 145), (315, 110)]}
    
    
    def __init__(self, level, dotList):
        self.level = level
        for d in Level.level_dict[self.level]:
            dot = Dot(d[0], d[1])
            dotList.add(dot)


### Draw functions

def drawStart(screen, font, instruction):
    startText = "The Ramsey Theory Game"
    insText = "Click Anywhere to Start"
    start = font.render(startText, True, BLACK)
    image = screen.blit(instruction ,(50, 50))
    ins = font.render(insText, True, BLACK)
    screen.blit(start, (screen.get_width() // 2 - start.get_width() // 2,\
                        start.get_height()))
    screen.blit(ins, (screen.get_width() // 2 - ins.get_width() // 2,\
                        screen.get_height() - ins.get_height() * 2))

def drawPlayerPage(screen, font):
    textName = font.render("Enter your name", True, BLACK)
    promptName1 = font.render("Player 1: ", True, BLACK)
    promptName2 = font.render("Player 2: ", True, BLACK)
    textLevel = font.render("Enter level from 1 to 3: ", True, BLACK)
    textMode = font.render('Enter mode "a" or "b" : ', True, BLACK)
    screen.blit(textName,  (screen.get_width() // 2 - textName.get_width() \
                            // 2, textName.get_height() * 2))
    screen.blit(promptName1, (screen.get_width()//5 - promptName1.get_width()\
                                // 2, promptName1.get_height() * 4))
    screen.blit(promptName2, (screen.get_width()//5 -  promptName2.get_width()\
                                //2, textName.get_height() * 6))
    screen.blit(textLevel, (screen.get_width() // 4 - textLevel.get_width()\
                        // 2, screen.get_height()//2 + textName.get_height()*2))
    screen.blit(textMode, (screen.get_width() // 4 * 3 - textMode.get_width()\
                        // 2, screen.get_height()//2 + textName.get_height()*2))

def drawPlayerNames(screen, font, name1, name2, level, mode1, mode2):
    name1 = font.render(name1, True, BLACK)
    name2 = font.render(name2, True, BLACK)
    level = font.render(level, True, BLACK)
    if mode1: mode = font.render("a", True, BLACK)
    if mode2: mode = font.render("b", True, BLACK)
    if not mode1 and not mode2: mode = font.render("", True, BLACK)
    screen.blit(name1, (screen.get_width() //3, name1.get_height()*4))
    screen.blit(name2, (screen.get_width() //3, name1.get_height()*6))
    screen.blit(level, (screen.get_width() // 4 - level.get_width() // 2,\
                        screen.get_height()//2 + level.get_height()*4))
    screen.blit(mode, (screen.get_width() // 4 * 3 - mode.get_width() // 2,\
                        screen.get_height()//2 + level.get_height()*4))

def drawSpot(screen, spotCount, font, name1, name2):
    width = 30
    name = font.render(name1, True, BLACK)
    if spotCount == 0:
        name = font.render(name1, True, BLACK)
        width = name.get_width()
    elif spotCount == 1:
        name = font.render(name2, True, BLACK)
        width = name.get_width()
    width = max(30, width)
        
    spots = [[screen.get_width() // 3, name.get_height() * 5], \
            [screen.get_width() // 3, name.get_height() * 7], \
            [screen.get_width() // 4 - width // 2, screen.get_height() // 2\
                + name.get_height() * 5],\
            [screen.get_width() // 4 * 3 - width // 2, \
                screen.get_height() // 2 + name.get_height() * 5]]
    
    recLoc = spots[spotCount]
    recLoc.extend([width,3])
    pygame.draw.rect(screen, BLACK, recLoc)
        
def drawReady(screen, font):
    text = font.render("Hit Enter to start", True, RED)
    screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2,\
                        screen.get_height() // 2))
    
def drawPlayerStats(screen, font, p1Turn):
    margin = 15
    name1 = font.render(Player.players[0].name, True, Player.players[0].color)
    name2 = font.render(Player.players[1].name, True, Player.players[1].color)
    
    if p1Turn:
        pygame.draw.rect(screen, GREEN, [margin, name1.get_height() * 4,\
                        name1.get_width() * 3 // 2, name1.get_height()], 3)
    else:
        pygame.draw.rect(screen, GREEN, [margin, name2.get_height() * 6,\
                        name2.get_width() * 3 // 2, name2.get_height()], 3)
        
    screen.blit(name1, (margin, name1.get_height()*4))
    screen.blit(name2, (margin, name1.get_height()*6))

def drawTriNum(screen, font, name1, name2, num1, num2):
    margin = 10
    p1Score = name1 + ": " + str(num1)
    p2Score = name2 + ": " + str(num2)
    p1 = font.render(p1Score, True, Player.players[0].color)
    p2 = font.render(p2Score, True, Player.players[1].color)
    
    screen.blit(p1, (margin, p1.get_height()*9))
    screen.blit(p2, (margin, p2.get_height()*10))
    

def drawTitle(screen, font, mode, level):
    levels = checkLv(int(level))
    levels = (levels[0]+1, levels[1]+1)
    if mode:
        title = "Level " + level + ": Avoid k_" + str(levels)
    else:
        title = "Level " + level + ": Create complete graphs"
    title = font.render(title, True, BLACK)
    screen.blit(title, (screen.get_width() //2- title.get_width() // 2,\
                        title.get_height()))

def drawWinMode1(screen, font, text):
    win = font.render(text, True, BLACK)
    pygame.draw.rect(screen, WHITE, [screen.get_width() // 2 - win.get_width()\
        // 2, screen.get_height() // 2 - win.get_height() // 2,\
        win.get_width(),win.get_height()])
    screen.blit(win, (screen.get_width() // 2 - win.get_width() // 2, \
        screen.get_height() // 2 - win.get_height() // 2))   

def drawHelp(screen, line):
    pygame.draw.line(screen, HINTYELLOW, [line[0][0], line[0][1]], \
                        [line[1][0], line[1][1]], 5)

def drawWinMode2(screen, font, text, p1Text, p2Text):
    win = font.render(text, True, BLACK)
    p1 = font.render(p1Text, True, BLACK)
    p2 = font.render(p2Text, True, BLACK)

    pygame.draw.rect(screen, WHITE, [screen.get_width() // 2 -\
        max(win.get_width(), p1.get_width(), p2.get_width()) // 2, \
        screen.get_height() // 2 - win.get_height() // 2, max(win.get_width(),\
        p1.get_width(), p2.get_width()), (win.get_height() * 3)])
    screen.blit(win, (screen.get_width() // 2 - win.get_width() // 2, \
        screen.get_height() // 2 - win.get_height() // 2))  
    screen.blit(p1, (screen.get_width() // 2 - p1.get_width() // 2, \
        screen.get_height() // 2 + p1.get_height() // 2)) 
    screen.blit(p2, (screen.get_width() // 2 - p2.get_width() // 2, \
        screen.get_height() // 2 + p2.get_height() // 2 * 3))  
        
def drawHelpButton(screen, font):
    margin = 10
    help = font.render("Hold h for a hint", True, BLACK)
    pygame.draw.rect(screen, WHITE, [margin, 
        screen.get_height() - help.get_height(), help.get_width(),\
        help.get_height()])
    screen.blit(help, (margin, screen.get_height() - help.get_height()))

### Helper functions
def checkHit(line):
    coll = 0
    for dot in Dot.dots:
        if abs(line.x0 - dot.x) < 20 and abs(line.y0 - dot.y) < 20:
            coll += 1
            x0, y0 = dot.x, dot.y
        elif abs(line.x1 - dot.x) < 20 and abs(line.y1 - dot.y) < 20:
            coll += 1
            x1, y1 = dot.x, dot.y
    if coll == 2:
        line.x0, line.y0, line.x1, line.y1 = x0, y0, x1, y1
        return True
    return False
    
def checkNew(line):
    isNew = True
    newLine = (line.x0, line.y0, line.x1, line.y1)
    oppositeLine = (line.x1, line.y1, line.x0, line.y0)
    if newLine in Line.linesDrawn or oppositeLine in Line.linesDrawn:
        isNew = False
    Line.linesDrawn.add((line.x0, line.y0, line.x1, line.y1))
    Line.linesDrawn.add((line.x1, line.y1, line.x0, line.y0))
    return isNew


def isTriangle(point1, point2, point3, dotDict):
    if point3 in dotDict and point3 != point1 and point3 != point2 and point3 in dotDict[point1]:
        return True
    return False

def permutations(lst): # algorithm outlined from 112 website
    if len(lst) == 0: return [[]]
    else:
        allPerm = []
        for subPerm in permutations(lst[1:]):
            for i in range(len(subPerm)+1):
                allPerm += [subPerm[:i] + [lst[0]] + subPerm[i:]]
        return allPerm
  
def allTri(lines):
    dotDict = getDotDict(lines)
    result = []
    for point1 in dotDict:
        for point2 in dotDict[point1]:
            for point3 in dotDict[point2]:
                if isTriangle(point1, point2, point3, dotDict):
                    allPerm = permutations([point1, point2, point3])
                    unique = True
                    for perm in allPerm:
                        if perm in result:
                            unique = False
                    if unique:
                        result.append([point1, point2, point3])
    return result

def getDotDict(lines):
    dotDict = dict()
    for line in lines:
        if (line.x0, line.y0) in dotDict:
            dotDict[(line.x0,line.y0)].add((line.x1, line.y1))
        else:
            dotDict[(line.x0,line.y0)] = {(line.x1, line.y1)}
             
        if (line.x1, line.y1) in dotDict:
            dotDict[(line.x1,line.y1)].add((line.x0, line.y0))
        else:
            dotDict[(line.x1,line.y1)] = {(line.x0, line.y0)}
    return dotDict

def getDotList(dotDict):
    dotList = []
    for dot in dotDict:
        dotList.append(dot)
    return dotList

def existComplete(lines, num):
    dotDict = getDotDict(lines)
    dotList = getDotList(dotDict)
    sol = backtracking(dotDict, num, [], 0, dotList)
    return sol

def isValid(dotDict, state, newDot):
    for dot in state:
        if newDot not in dotDict[dot]:
            return False
    return True

def permutationsOfN(lst, n):
    result = []
    for perm in permutations(lst):
        if perm not in result:
            result.append(perm[:n])
    return result

# exist a complete graph based on # of vertices and edges
def checkExist(lines, num):
    dotDict = getDotDict(lines)
    dotList = getDotList(dotDict)
    allDot = permutationsOfN(dotList, num)
    connectedLines = 0
    for startpt in allDot:
        index = allDot.index(startpt)
        for endpt in allDot[index+1:]:
            for line in lines:
                if startpt[0] == line.x0 and startpt[1] == line.y0 and endpt[0] == line.x1 and endpt[1] == line.y1:
                    connectedLines += 1
    return num*(num-1)/2 == connectedLines
    
# backtrack: take in the dictionary and the num of Kn, move by dots
def backtracking(dotDict, num, state, index, dotList):
    if len(state) == num:
        return state
    else:
        for dot in dotList[index+1:]:
            nextConnect = dotDict[dot]
            if len(nextConnect) >= num:
                if isValid(dotDict, state, dotList[index+1]):
                    state.append(dotList[index+1])
                    temp = backtracking(dotDict, num, state, index+1, dotList)
                    if temp != None:
                        return temp
                    state.pop()
        return None


def checkLv(lv):
    if lv == 1:
        return (2, 2)
    elif lv == 2:
        return (2, 3)
    else:
        return (3, 3)

def finished(p1Lines, p2Lines):
    total = len(p1Lines) + len(p2Lines)
    # for n number of nodes ,There are n*(n-1)/2 edges
    nodes = len(Dot.dots)
    return (nodes*(nodes-1)/2 == total)

def checkHigher(triSol, lines):
    checking = 2
    higher = ["currently triangles"]
    highest = False
    while not highest:
        higher = existComplete(lines, checking + 1)
        if higher == None:
            highest = True
        else:
            checking += 1
    return checking
            

def getList(lines):
    result = []
    for line in lines:
        result.append(((line.x0, line.y0), (line.x1, line.y1))) # double tuple
    return result

# hints return tuples that contains line coordinates
def getHintMode1(p1Lines, p2Lines):
    temp = p1Lines.copy()
    score = len(allTri(p1Lines))
    p1Lst = getList(p1Lines)
    p2Lst = getList(p2Lines)
    legalLines = legalMoves(p1Lst, p2Lst)
    for line in legalLines:
        newLine = Line(line[0][0], line[0][1], line[1][0], line[1][1], HINTYELLOW)
        temp.add(newLine)
        newScore = len(allTri(temp))
        if newScore <= score:
            return line
        temp.remove(newLine)
    return legalLines[0]

def getHintMode2(p1Lines, p2Lines):
    temp = p1Lines.copy()
    enemyTemp = p2Lines.copy()
    score = len(allTri(p1Lines))
    highest = checkHigher(allTri(p1Lines), p1Lines)
    enemyScore = len(allTri(p2Lines))
    enemyHighest = checkHigher(allTri(p2Lines), p2Lines)
    p1Lst = getList(p1Lines)
    p2Lst = getList(p2Lines)
    legalLines = legalMoves(p1Lst, p2Lst)
    
    for line in legalLines:
        newLine = Line(line[0][0], line[0][1], line[1][0], \
                        line[1][1], HINTYELLOW)
        temp.add(newLine)
        newScore = len(allTri(temp))
        newHighest = checkHigher(allTri(temp), p1Lines)
        if newHighest > highest:
            return line
        temp.remove(newLine)
        
    for line in legalLines:
        newLine = Line(line[0][0], line[0][1], line[1][0], \
                        line[1][1], HINTYELLOW)
        temp.add(newLine)
        newScore = len(allTri(temp))
        newHighest = checkHigher(allTri(temp), p1Lines)
        if newScore > score:
            return line
        temp.remove(newLine)
        
    for line in legalLines:
        newLine = Line(line[0][0], line[0][1], line[1][0], \
                        line[1][1], HINTYELLOW)
        enemyTemp.add(newLine)
        newScore = len(allTri(enemyTemp))
        newHighest = checkHigher(allTri(enemyTemp), p2Lines)
        if newHighest > enemyHighest:
            return line
        enemyTemp.remove(newLine)
    
    for line in legalLines:
        newLine = Line(line[0][0], line[0][1], line[1][0], \
                        line[1][1], HINTYELLOW)
        enemyTemp.add(newLine)
        newScore = len(allTri(enemyTemp))
        newHighest = checkHigher(allTri(enemyTemp), p2Lines)
        if newScore > enemyScore:
            return line
        enemyTemp.remove(newLine)
    
    return legalLines[0]
    
def minimaxMove(p1Lines, p2Lines, p1Win, p2Win): # algorithm sourced from 112 website
    if finished(p1Lines, p2Lines):
        return (None, inf) if p1Win else (None, -inf)
    else:
        bestMove = None
        bestScore = -inf
        for move in legalMoves(p1Lines, p2Lines):
            makeMove(p1Lines, move, p1)
            _, moveScore = minimaxMove(p2Lines, p1Lines, p2Win, p1Win)
            undoMove(p1Lines, move)
            if moveScore > bestScore:
                bestScore = moveScore
                bestMove = move
        return (bestMove, bestScore)

def legalMoves(p1Lst, p2Lst):
    allLines = getAllLines()
    possibleLines = []
    for line in allLines:
        if line not in p1Lst and line not in p2Lst:
            opposite = (line[1], line[0])
            if opposite not in p1Lst and opposite not in p2Lst:
                possibleLines.append(line)
    return possibleLines

def getAllLines():
    allLines = []
    for dot in range(len(Dot.dots)):
        for next in range(len(Dot.dots)):
            if (Dot.dots[dot].x, Dot.dots[dot].y) != (Dot.dots[next].x,\
                Dot.dots[next].y):
                allLines.append(((Dot.dots[dot].x, Dot.dots[dot].y),\
                    (Dot.dots[next].x, Dot.dots[next].y))) # double tuple
    return allLines

def makeMove(lines, move, player):
    newLine = Line(move[0][0], move[0][1], move[1][0], move[1][1], player.color)
    lines.add(newLine)

def undoMove(lines, move, player):
    newLine = Line(move[0][0], move[0][1], move[1][0], move[1][1], player.color)
    line.remove(newLine)


