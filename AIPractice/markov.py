import numpy as np 

states = ['Sunny', 'Cloudy', 'Rainy']
transition_matrix = np.array([[0.5, 0.2, 0.3],
                             [0.3, 0.6, 0.1],
                             [0.4, 0.1, 0.5]])



def simulate_markov_process(initial_state, num_steps):
    current_state = initial_state
    state_sequnce = [current_state]
    rainy =0
    for _ in range(num_steps):
        if current_state == 'Sunny':
            next_state = np.random.choice(states, p=transition_matrix[0])
        elif current_state == 'Cloudy':
            next_state = np.random.choice(states, p=transition_matrix[1])
        else:
             next_state = np.random.choice(states, p=transition_matrix[2])
        
        state_sequnce.append(next_state)
        if next_state == 'Rainy':
            rainy+=1
        current_state = next_state
    
    return state_sequnce, rainy

initial_state = 'Sunny'
num_steps = 10
state_sequnce, rainy = simulate_markov_process(initial_state, num_steps)

print(f'State seuence for {num_steps} steps starting from {initial_state}')
print("->".join(state_sequnce))

count_at_least_3_rainy = 0
for _ in range (1000):
    _, rainy = simulate_markov_process(initial_state, num_steps)
    if rainy >= 3:
        count_at_least_3_rainy+=1
print ("\n probablity of rainy day is: ", count_at_least_3_rainy/1000)