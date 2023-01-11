# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:45:31 2023

@author: dylan
"""

import numpy as np

mapsize = 81
goal = 80
gamma = 0.9

isObstacle = np.zeros((81))
current_state = np.zeros((2)).astype(int)
next_state = np.zeros((2)).astype(int)

R = np.full((81,81),-10)

def setR():
    #R[A][B]:從a移到b的reward value
    for x in range (0,81):
        R[x][x] = -2
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
                R[x-9][x]=-10
                R[x-1][x]=-10
                R[x+1][x]=-10
            elif(x // 9 == 0):#下排
                R[x+9][x]=-10
                R[x-1][x]=-10
                R[x+1][x]=-10
            else:
                R[x+9][x]=-10
                R[x-9][x]=-10
                R[x-1][x]=-10
                R[x+1][x]=-10
            isObstacle[x]=1
        ##橫排障礙
        elif(x//9==3 and (x%9>0 and x%9<8)):
            R[x+9][x]=-10
            R[x-9][x]=-10
            R[x-1][x]=-10
            R[x+1][x]=-10
            isObstacle[x]=1
        elif(x//9==5 and (x%9>0 and x%9<8)):
            R[x+9][x]=-10
            R[x-9][x]=-10
            R[x-1][x]=-10
            R[x+1][x]=-10
            isObstacle[x]=1
        #能到終點的給100分reward
        if(R[x][80]==0):
            R[x][80]=100
        R[80][80]=200
        isObstacle[80] = 2

def RL_environment(current_state,action,reset):
    
    '''
    if(current_state[0] == None):
        current_state[0] = random.randint(0,80)
    if(current_state[1] == None):
        current_state[1] = random.randint(0,80)
    '''
    if(current_state[0] == None and current_state[1] == None):
        current_state = np.random.choice(81, 2, replace=False)
        #print('current',current_state)
    

    reward = np.zeros((2))
    if(current_state[0] == current_state[1]):
        if(current_state[0] == 80 and current_state[1] == 80):
            reward[0] = 1000
            reward[1] = 1000
        else:
            reward[0] = -100
            reward[1] = -100
    else:
        reward[0] = R[current_state[0]][action[0]]
        reward[1] = R[current_state[1]][action[1]]
        #print(current_state[0],action[0],reward[0])
    if reset == True:
        current_state = np.random.choice(81, 2, replace=False)
        reward[0] = -10
        reward[1] = -10
        
    return reward,current_state

def train():
    
    goal_state = 80
    episodes = 10

    #Q[agent][current_state][possible_action]
    Q = np.zeros((2,81,81)).astype(int)

    #learn from each iterations
    for i in range(episodes):
        valid_action_on_state = [-10,-10]
        #reset the environment
        _, current_state = RL_environment([None,None],[1,1],True)
        while(True):
            while(valid_action_on_state[0] == -10 or valid_action_on_state[1] == -10):
                possible_action = np.random.choice(81, 2)
                #print(possible_action)
                reward = np.zeros((2))
                reward,_ = RL_environment(current_state,possible_action,False)#send in 2 states, 2 actions and reset flag
                
                valid_action_on_state[0] = reward[0]
                valid_action_on_state[1] = reward[1]
            print('current episode :',i,'  ',current_state,'to',possible_action,' is valid!',reward)    
            valid_action_on_state[0] = valid_action_on_state[1] = -10
            
            next_state[0] = possible_action[0]
            next_state[1] = possible_action[1]
            q0Max = Q[0].max(axis=1)
            q1Max = Q[1].max(axis=1)
            Q[0][current_state[0]][possible_action[0]] = reward[0] + (gamma * q0Max[next_state[0]])
            Q[1][current_state[1]][possible_action[1]] = reward[1] + (gamma * q1Max[next_state[1]])
            
            #print(q0Max,q1Max)
            #print(current_state[0],current_state[1])
            #if(current_state[0] == current_state[1]):
                #print("collision!!!")
            #print('--------')
            if(current_state[0] == goal_state and current_state[1] == goal_state):
                break
            current_state[0] = possible_action[0]
            current_state[1] = possible_action[1]
        print('finished episode %d restart environment'%(i))
    return Q

def test(Q):
    possible_initial_states = np.zeros((2)).astype(int)
    possible_initial_states[0] = 18
    possible_initial_states[1] = 54
    print('initial room agent 1 :',possible_initial_states[0])
    print('initial room agent 2 :',possible_initial_states[1])
    goal_state = 80

    action_max0 = np.argmax(Q[0],axis=1)
    action_max1 = np.argmax(Q[1],axis=1)
    #print(action_max0[possible_initial_states[0]],action_max1[possible_initial_states[1]])
    msg = ''
    
    for i in range(1):
        curr_state = np.zeros((2)).astype(int)
        curr_state[0] = possible_initial_states[0]
        curr_state[1] = possible_initial_states[1]
        
        msg = ('initial state [agent 1,agent 2] = [%d %d] -> = ['%(curr_state[0],curr_state[1]))
        while(True):
            next_state[0] = action_max0[curr_state[0]]
            next_state[1] = action_max1[curr_state[1]]
            msg += (' -> [%d %d] '%(next_state[0],next_state[1]))
            #print(curr_state)
            curr_state[0] = next_state[0]
            curr_state[1] = next_state[1]
            if(curr_state[0] == goal_state and curr_state[1] == goal_state):
                msg+=(']')
                print(msg)
                break
        print('\n')

setR()
Q = train()
print(Q)
test(Q)



















