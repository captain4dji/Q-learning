import random
import numpy as np

isObstacle = np.zeros((81))
mapsize = 81
goal = 80
reward = np.zeros((2))
gamma = 0.9

def setMap(mapsize,R,current_agent):
    for x in range (0,mapsize):
        if(x%9==0 and x//9==0):#左下
            R[current_agent][x][x+1]=0
            R[current_agent][x][x+9]=0
        elif(x%9==0 and x//9==8):#左上
            R[current_agent][x][x+1]=0
            R[current_agent][x][x-9]=0
        elif(x%9==8 and x//9==0):#右下
            R[current_agent][x][x-1]=0
            R[current_agent][x][x+9]=0
        elif(x%9==8 and x//9==8):#右上
            R[current_agent][x][x-1]=0
            R[current_agent][x][x-9]=0
        elif(x % 9 == 0):#左排
            R[current_agent][x][x-9]=0
            R[current_agent][x][x+9]=0
            R[current_agent][x][x+1]=0
        elif(x % 9 == 8):#右排
            R[current_agent][x][x-9]=0
            R[current_agent][x][x+9]=0
            R[current_agent][x][x-1]=0
        elif(x // 9 == 0):#下排
            R[current_agent][x][x+9]=0
            R[current_agent][x][x-1]=0
            R[current_agent][x][x+1]=0
        elif(x // 9 == 8):#上排
            R[current_agent][x][x-9]=0
            R[current_agent][x][x-1]=0
            R[current_agent][x][x+1]=0
        else:
            R[current_agent][x][x+9]=0
            R[current_agent][x][x-9]=0
            R[current_agent][x][x-1]=0
            R[current_agent][x][x+1]=0
        
    for x in range (0,mapsize):
        #障礙(無法移到x)
        ##直排障礙
        if(x%9==4 and x//9!=4):
            if(x // 9 == 8):#上排
                R[current_agent][x-9][x]=-1
                R[current_agent][x-1][x]=-1
                R[current_agent][x+1][x]=-1
            elif(x // 9 == 0):#下排
                R[current_agent][x+9][x]=-1
                R[current_agent][x-1][x]=-1
                R[current_agent][x+1][x]=-1   
            else:
                R[current_agent][x+9][x]=-1
                R[current_agent][x-9][x]=-1
                R[current_agent][x-1][x]=-1
                R[current_agent][x+1][x]=-1
            isObstacle[x]=1
        ##橫排障礙
        if((x//9==3 or x//9==5) and (x%9>0 and x%9<8)):
            R[current_agent][x+9][x]=-1
            R[current_agent][x-9][x]=-1
            R[current_agent][x-1][x]=-1
            R[current_agent][x+1][x]=-1
            isObstacle[x]=1
        
        #能到終點的給100分reward
        if(R[current_agent][x][goal]==0):
            R[current_agent][x][goal]=100
        R[current_agent][goal][goal]=100
        isObstacle[goal] = 2

def RL_environment(current_state,action,reset,current_agent):
    if(current_state == None):
       current_state = random.randint(0,mapsize-1)
    R = np.full((2,mapsize,mapsize),-1)
    #R[x][a][b]:agent x從a移到b的reward value
    setMap(mapsize,R,current_agent)

    reward[current_agent] = R[current_agent][current_state][action]

    if reset == True:
        current_state = random.randint(0,mapsize-1)
        reward[current_agent] = -1
    
    return reward[current_agent],current_state


def train():
    
    goal_state = goal
    episodes = 20

    Q = np.zeros((2,mapsize,mapsize)).astype(int)

    #learn from each iterations
    for i in range(episodes):
        valid_action_on_state = -1
    
        for agent in range(2):
        #reset the environment
            _, current_state = RL_environment(None,1,True,agent)
            while(True):
                while(valid_action_on_state == -1):
                    possible_action = random.randint(0,mapsize-1)
                    reward[agent],_ = RL_environment(current_state,possible_action,False,agent)
                    valid_action_on_state = reward[agent]
                valid_action_on_state = -1
                next_state = possible_action
                qMax = Q[agent].max(axis=1)
                Q[agent][current_state][possible_action] = reward[agent] + (gamma * qMax[next_state])
                if(current_state == goal_state):
                    break

                current_state = possible_action
        print('finished episode %d restart environment'%(i))
    return Q

def test(Q):
    possible_initial_states = [[0],[5]]
    goal_state = 80
    for agent in range(2):
        note = ('agent %d starting positions: '%(agent))
        for i in range(len(possible_initial_states[agent])):
            note += str(possible_initial_states[agent][i])
            note += (' ')
        print(note)
        
    
        action_max = np.argmax(Q[agent],axis=1)
        msg = ''
    
        for i in range(len(possible_initial_states[agent])):
            curr_state = possible_initial_states[agent][i] 
            msg = ('initial room %d -> = ['%(curr_state))
            #print(curr_state,action_max[agent,curr_state])
            while(True):
                next_state = action_max[curr_state]
                #print(curr_state,next_state,goal_state)
                msg += (' -> %d '%(next_state))
                curr_state = next_state
                if(curr_state == goal_state):
                    msg+=(']')
                    print(msg)
                    break
            print('\n')
        
        
Q = train()
print("finished training")
print(Q)
test(Q)

print('gamma = %f'%(gamma))
print("map : ")
for x in range (0,mapsize):
    if(isObstacle[x]==1):
        print('|%2d'%x,end = '')
    elif(isObstacle[x]==2):
        print('|*G',end='')
    else:
        print('|..',end = '')
    if(x%9 == 8):
        print("|\r")
        print("----------------------------")
