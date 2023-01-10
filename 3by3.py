import random
import numpy as np

def RL_environment(current_state,action,reset):
    if(current_state == None):
       current_state = random.randint(0,8)
    
    R = [[-1,0,-1,0,-1,-1,-1,-1,-1],
         [0,-1,0,-1,0,-1,-1,-1,-1],
         [-1,0,-1,-1,-1,0,-1,-1,-1],
         [0,-1,-1,-1,0,-1,0,-1,-1],
         [-1,0,-1,0,-1,0,-1,0,-1],
         [-1,-1,0,-1,0,-1,-1,-1,100],
         [-1,-1,-1,0,-1,-1,-1,0,-1],
         [-1,-1,-1,-1,0,-1,0,-1,100],
         [-1,-1,-1,-1,-1,0,-1,0,100]]

    reward = R[current_state][action]

    if reset == True:
        current_state = random.randint(0,8)
        reward = -1
    
    return reward,current_state

gamma = 0.9

def train():
    
    goal_state = 8
    episodes = 20

    Q = np.zeros((9,9)).astype(int)

    #learn from each iterations
    for i in range(episodes):
        valid_action_on_state = -1
    
        #reset the environment
        _, current_state = RL_environment(None,1,True)

        while(True):
            while(valid_action_on_state == -1):
                possible_action = random.randint(0,8)
                reward,_ = RL_environment(current_state,possible_action,False)
                valid_action_on_state = reward
            valid_action_on_state = -1
            next_state = possible_action
            qMax = Q.max(axis=1)
            #print(Q)
            #
            Q[current_state][possible_action] = reward + (gamma * qMax[next_state])
            if(current_state == goal_state):
                break

            current_state = possible_action
        print('finished episode %d restart environment'%(i))
        #input()
    return Q

def test(Q):
    possible_initial_states = np.random.permutation(8)
    print(possible_initial_states)
    goal_state = 8

    action_max = np.argmax(Q,axis=1)
    msg = ''

    for i in range(len(possible_initial_states)):
        curr_state = possible_initial_states[i] 
        msg = ('initial state room %d actions = ['%(curr_state))
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