import pygame
import random
import numpy


M=10
N=10
grid_size = 25
population_size = 500
parents_size = 50

def Snake_Game():
    global Snake, screen, loop, pause_time
    pygame.init()
    screen = pygame.display.set_mode(M*grid_size, N*grid_size)
    (x1,y1) = random.choice(grids)
    (x2,y2) = random.choice([(x1+1,y1),(x1,y1+1),(x1-1,y1),(x1,y1-1)])
    Snake = [(x1,y1),(x2,y2)]
    steps = 0
    pause_time = 60
    food()
    loop = True
    while loop:
        steps = steps+1
        prediction_from_genetic_weights()
        update_snake()
        if len(Snake) == N*N:
            print('Congratulations! Your Snake won')
            loop = False
        elif snake_head == Food:
            food()
            #Snake.append(snake_tail)
        elif snake_head not in grids or snake_head in snake_body:
            loop = False
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                pygame.quit()
                loop = False
            elif event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_SPACE):
                    if(pause_time == 60):
                        pause_time = 0
                    else:
                        pause_time = 60
        #if(Snake[0], Food) 
    score = len(Snake) - 2
    return (score+0.5+0.5*(score-steps/(score+1))/(score+steps/(score+1)))*10000, score, steps

def food():
    global Food
    snake_no_grids = [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)

def prediction_from_genetic_weights():
    global action
    dirc = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    temp = [[0,1], [1,0], [0,-1], [-1,0]]
    head_dirc = dirc[temp.index((Snake[0][0] - Snake[1][0], Snake[0][1] - Snake[1][1]))]
    (x,y) = Snake[0]
    d1 = [(x,_) for _ in range(y+1, N)]
    d2 = [(x+_,y+_) for _ in range(1,min(M-x,N-y))]
    d3 = [(_,y) for _ in range(x+1, M)]
    d4 = [(x+_,y-_) for _ in range(1,max(M-x,N-y)) if (x+_,y-_) in grids]
    d5 = [(x,_) for _ in range(0, y)]
    d6 = [(x-_,y-_) for _ in range(1,min(x,y)+1)]
    d7 = [(_,y) for _ in range(0, x)]
    d8 = [(x-_,y+_) for _ in range(1,max(M-x,N-y))  if (x-_,y+_) in grids]
    d = [d1, d2,d3,d4,d5,d6,d7,d8]
    val = min(M,N) - 1
    wall_distance = [len(i)/val for i in d]
    food_presence = [(val - j.index(Food))/val if Food in j else 0 for j in d]
    body_presence = []
    vision = 
    input_layer = vision + head_dirc
    action = neural_network(input_layer)

def neural_network(ip):
    m1, s1 = numpy.reshape(ip, (1,NN[0])), 0


def relu(x):
    lst=[_ if _>0 else 0 for _ in x[0]]
    return numpy.array(lst)

def sigmoid(x):
    lst=[1/(1+(numpy.exp(-1*_))) for _ in x[0]]
    return numpy.array(lst)

def update_snake():
    global snake_tail, snake_head, snake_body
    display()
    pygame.time.wait(pause_time)
    (x,y) = Snake[0]
    if(action == 'Top'):
        Snake.insert(0,(x,y+1))
    elif (action == 'Right'):
        Snake.insert(0,(x+1, y))
    elif (action == 'Bottom'):
        Snake.insert(0,(x,y-1))
    else:
        Snake.insert(0,(x-1,y))
    snake_tail = Snake.pop()
    snake_head, snake_body = Snake[0], Snake[1:]


def display():
    pygame.draw.rect(screen, (0,0,0), (0,0,M*grid_size,N*grid_size))
    pygame.draw.rect(screen, (255,255,255), (snake_head[0]*grid_size, snake_head[1].grid_size, grid_size, grid_size))
    [pygame.draw.rect(screen, (255,255,255), (i[0]*grid_size, i[1]*grid_size, grid_size, grid_size)) for i in snake_body]
    pygame.draw.rect(screen, (0,255,0), (Food[0]*grid_size, Food[0]*grid_size, grid_size, grid_size))
    pygame.display.update()

def crossver():
    global offspring
    offspring = []
    for _ in range(population_size - parents_size):
        parent1_id = random.choice(Roulette_wheel)
        parent2_id = random.choice(Roulette_wheel)
        while parent1_id == parent2_id:
            parent1_id = random.choice(Roulette_wheel)
        for i in range(weights_length):
            if(random.uniform(0,1) < 0.5):
                wts = [parents[parents1_id][i]]
            else:
                wts = [parents[parents2_id][i]]
        offspring.append(wts)

def mutation():
    global offspring
    for i in range(population_size-parents_size):
        for _ in range(int(weights_length*0.05)):
            j = random.randint(0,weights_length-1)
            value = random.choice(numpy.arange(-0.5,0.5,step=0.001))
            offspring[i][j] = offspring[i][j] + value

NN = [28,8,4]
AF = [relu, sigmoid]
pause_time, generation_length, mloop = 0,2000,True
grids = [(i,j) for i in range(M) for j in range(N)]
Actions = ['Top', 'Right', 'Bottom', 'Left']
Roulette_wheel=list(range(0, int(0.2*parents_size))*3 + list(range(int(0.2*parents_size), int(0.5*parents_size)))*2 + list(range(int(0.5*parents_size), parents_size))
weights_length=sum([NN[_]*NN[_+1]+NN[_+1] for _ in range(len(NN)-1)])
population = numpy.random.choice(numpy.arange(-1,1,step = 0.001), size = (population_size, weights_length), replace = True)
while Generation < generation_length and mloop:
    print('###################### ','Generation ',Generation,' ######################')
    Fitness = []
    Score = []
    i = 0
    while(i<population_size and mloop):
        weigths = list(population[i,:])
        i = i+1
        fitness, score, steps = Snake_game()
        print('Chromosome ',i,' >>> ','Score : ',score,', Steps : ',steps,', Fitness : ',fitness)
        Fitness.append(fitness), Score.append(score)
    parents = []
    max_fitness = max(Fitness)
    avg_score = sum(Score)/generation_length
    j = 0
    while j<parents_size and mloop:
        j = j+1
        parents_id = Fitness.index(max(Fitness))
        Fitness[parens_id] = -999
        parents.append(list(population[parents_id, :]))
    while mloop and j==parents.length:
        j = j+1
        High_score = max(max(Score), High_Score)
        print('Generation high score : ',max(Score),', Generation Avg score : ',avg_score,', Overall high score : ',High_score)
        crossover()
        mutation()
        population = numpy.reshape(parents+offspring, (population_size, -1))
    Generation = Generation +1
pygame.quit()