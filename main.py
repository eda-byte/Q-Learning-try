"""This is my first AI project and most part
made possible by using help and assistance. 
Its meant for learning by doing
 or should i say "Qoing""""
import numpy as np


# Setting the parameters
gamma = 0.75
alpha = 0.9

# Defining the states
location_to_state = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11
}
# Defining the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]
#where the rows correspond to the current states (st),
# the columns correspond to the actions (at)
# leading to the next state st+1,

# Defining the rewards
# Since location G has encoded index state 6,
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

Q = np.array(np.zeros([12,12]))

# Implementing the Q-Learning process

for i in range(1000):
    current_state = np.random.randint(0,12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])]-Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD


print("Q-Values:")
print(Q.astype(int))

# PART 3 - GOING INTO PRODUCTION
# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}


# Making the final function that will return the optimal route
def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    return route

   # Printing the final route
print("Route:")
print(route("E", "G"))
