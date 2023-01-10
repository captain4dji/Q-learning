import random
import numpy as np

isObstacle = np.zeros((81))
gamma = 0.9

def RL_environment(current_state,action,reset):
    if(current_state == None):
       current_state = random.randint(0,80)
    R = np.full((81,81),-1)
    #R[A][B]:從a移到b的reward value
    for x in range (0,81):
        if(x%9==0 and x//9==0):#左下
            R[x][x+1]=0
            R[x][x+9]=0
        elif(x%9==0 and x//9==8):#左上
            R[x][x+1]=0
            R[x][x-9]=0
        elif(x%9==8 and x//9==0):#右下
            R[x][x-1]=0
            R[x][x+9]=0
        elif(x%9==8 and x//9==8):#右上
            R[x][x-1]=0
            R[x][x-9]=0
        elif(x % 9 == 0):#左排
            R[x][x-9]=0
            R[x][x+9]=0
            R[x][x+1]=0
        elif(x % 9 == 8):#右排
            R[x][x-9]=0
            R[x][x+9]=0
            R[x][x-1]=0
        elif(x // 9 == 0):#下排
            R[x][x+9]=0
            R[x][x-1]=0
            R[x][x+1]=0
        elif(x // 9 == 8):#上排
            R[x][x-9]=0
            R[x][x-1]=0
            R[x][x+1]=0
        else:
            R[x][x+9]=0
            R[x][x-9]=0
            R[x][x-1]=0
            R[x][x+1]=0
    for x in range(0,81):
        #障礙(無法移到x)
        ##直排障礙
        if(x%9==4 and x//9!=4):
            if(x // 9 == 8):#上排
                R[x-9][x]=-1
                R[x-1][x]=-1
                R[x+1][x]=-1
            elif(x // 9 == 0):#下排
                R[x+9][x]=-1
                R[x-1][x]=-1
                R[x+1][x]=-1   
            else:
                R[x+9][x]=-1
                R[x-9][x]=-1
                R[x-1][x]=-1
                R[x+1][x]=-1
            isObstacle[x]=1
        ##橫排障礙
        elif(x//9==3 and (x%9>0 and x%9<8)):
            R[x+9][x]=-1
            R[x-9][x]=-1
            R[x-1][x]=-1
            R[x+1][x]=-1
            isObstacle[x]=1
        elif(x//9==5 and (x%9>0 and x%9<8)):
            R[x+9][x]=-1
            R[x-9][x]=-1
            R[x-1][x]=-1
            R[x+1][x]=-1
            isObstacle[x]=1
        #能到終點的給100分reward
        if(R[x][80]==0):
            R[x][80]=100
        R[80][80]=100
        isObstacle[80] = 2

    
    reward = R[current_state][action]
    if reset == True:
        current_state = random.randint(0,80)
        reward = -1
    return reward,current_state

def train():
    
    goal_state = 80
    episodes = 20

    Q = np.zeros((81,81)).astype(int)

    #learn from each iterations
    for i in range(episodes):
        valid_action_on_state = -1
        #reset the environment
        _, current_state = RL_environment(None,1,True)
        while(True):
            while(valid_action_on_state == -1):
                possible_action = random.randint(0,80)
                reward,_ = RL_environment(current_state,possible_action,False)
                valid_action_on_state = reward
            valid_action_on_state = -1
            next_state = possible_action
            qMax = Q.max(axis=1)
            Q[current_state][possible_action] = reward + (gamma * qMax[next_state])
            if(current_state == goal_state):
                break
            current_state = possible_action
        print('finished episode %d restart environment'%(i))
    return Q

def test(Q):
    possible_initial_states = [0,21,23,57,59]
    print('test room :',possible_initial_states)
    goal_state = 80

    action_max = np.argmax(Q,axis=1)
    msg = ''
    
    for i in range(len(possible_initial_states)):
        curr_state = possible_initial_states[i] 
        msg = ('initial state room %d -> = ['%(curr_state))
        while(True):
            next_state = action_max[curr_state]
            msg += (' -> %d '%(next_state))
            
            curr_state = next_state
            if(curr_state == goal_state):
                msg+=(']')
                print(msg)
                break
        print('\n')

Q = train()
print(Q)
test(Q)
