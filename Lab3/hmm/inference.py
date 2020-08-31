import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                    all_possible_observed_states,
                    prior_distribution,
                    transition_model,
                    observation_model,
                    observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 

    # TODO: Compute the forward messages
    alpha_0 = rover.Distribution() # initial forward message

    for z in prior_distribution:
        alpha_0[z] = prior_distribution[z]*observation_model(z)[observations[0]] # alpha(z_0) = p(z_0)*p((x_0, y_0)|z_0)

    alpha_0.renormalize()

    forward_messages[0] = alpha_0

    # compute all the other forward messages (alphas)
    for i in range(num_time_steps - 1):
        curr_observation = observations[i+1] 
        alpha = rover.Distribution() # initialize the current forward message before filling in the values
        
        # go through all values for z_n to fill it in for the current forward message
        for z_n in all_possible_hidden_states:
            # account for possibilility of missing data
            if curr_observation == None:
                cond_prob = 1
            else:
                cond_prob = observation_model(z_n)[curr_observation] 

            #  if the condition probability isn't 0, we need to calculate the value for alpha(z_n), otherwise we can skip it and automatically set the value to 0
            if cond_prob != 0:
                prev_sum = sum(forward_messages[i][prev_z]*transition_model(prev_z)[z_n] for prev_z in forward_messages[i]) 
                alpha[z_n] = cond_prob* prev_sum # alpha(z_n) = p((x_n, y_n)|z_n)*sum of alpha(z_{n - 1})*p(z_n|z_{n - 1}) over all possible z_{n - 1} values, calculated in previous line
            else:
                alpha[z_n] = 0
        
        alpha.renormalize() # renormalize the values into a probability distribution, in order to prevent integer underflow
        forward_messages[i+1] = alpha # save the alpha value for this timestep
    
    # TODO: Compute the backward messages
    beta_n = rover.Distribution() # initial backwards message (for final state)

    # we don't have information, so just set everything to be 1 in order not to bias the messages that build upon this
    for state in all_possible_hidden_states:
        beta_n[state] = 1
    beta_n.renormalize()

    backward_messages[num_time_steps - 1] = beta_n

    #  compute other backwards messages
    for i in reversed(range(num_time_steps - 1)):
        curr_observation = observations[i + 1]
        beta = rover.Distribution() # initialize the current backwards message before filling in its values

        # go through all values of z_{n - 1} to fill in for the current backwards message
        for prev_z in all_possible_hidden_states:
            trans_model = transition_model(prev_z) # used to find conditional probabilities of the next state conditioned on z_{n - 1}, precomputed to reduce runtime
            
            # if current observation is none, we need to replace the p((x_n, y_n)|z_n) factor with 1, as we don't have information and don't want to bias the predictions
            # otherwise, we can include that factor in our calculation
            if curr_observation == None:
                # beta(z_{n - 1}) = sum of beta(z_n)*p(z_n|z_{n - 1}) over all possible z_n values from the last backwards message
                beta[prev_z] = sum(backward_messages[i + 1][z_n]*trans_model[z_n] for z_n in backward_messages[i+1])
            else:
                # beta(z_{n - 1}) = sum of beta(z_n)*p((x_n, y_n)|z_n)*p(z_n|z_{n - 1}) over all possible z_n values from the last backwards message
                beta[prev_z] = sum(backward_messages[i + 1][z_n]*observation_model(z_n)[curr_observation]*trans_model[z_n] for z_n in backward_messages[i+1])

        beta.renormalize() # renormalize the values into a probability distribution to prevent integer underflow
        backward_messages[i] = beta # save the beta value for this timestep
    
    # TODO: Compute the marginals 

    # to compute the marginals, we need to compute every gamma(z_n) and then normalize it
    for i in range(num_time_steps):
        alpha = forward_messages[i]
        beta = backward_messages[i]
        gamma = rover.Distribution()

        for state in all_possible_hidden_states:
            gamma[state] = alpha[state]*beta[state] # gamma(z_n) = alpha(z_n)*beta(z_n), for each possible state

        gamma.renormalize() # renormalize to make sure it is a probability distribution
        marginals[i] = gamma # save the marginal distribution for this timestep
        
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps # used to store all w values
    max_states = [None] * num_time_steps # used to store the state that maximized the corresponding w value
    estimated_hidden_states = [None] * num_time_steps # used to store the final computed maximum likelihood estimate of the sequences of states

    # initialize the first w value
    w_0 = dict() # using regular dicts since for viterbi we are working in the log domain, so they are no longer probability distributions
    
    for z in prior_distribution:
        w_0[z] = np.log(prior_distribution[z]*observation_model(z)[observations[0]]) # w_1(z_1) = ln(p(x_1|z_1)*p(z_1)) for each possible state

    w[0] = w_0 # save the initial w value

    # compute all the other w values
    for i in range(num_time_steps - 1):
        curr_observation = observations[i+1]
        next_w = dict() # a dict to keep track of the current w values for each state
        path_tracer = dict() # a dict to keep track of which previous state maximixes the w value for the current state

        # go through all values of z_{n + 1} to fill in the value for w_{n + 1}
        for next_z in all_possible_hidden_states:
            # accounting for possibility of missing observations
            if curr_observation == None:
                cond_prob = 1
            else:
                cond_prob = observation_model(next_z)[curr_observation] 

            # if the conditional prob is 0, we don't need to bother computing the value since it just becomes ln(0) 
            # otherwise, compute the w value using the recursive equation and keep track of the state that maximixed it
            if cond_prob != 0:
                dist = {z_n:(np.log(transition_model(z_n)[next_z]) + w[i][z_n]) for z_n in w[i].keys()} # create a dict that stores all values of ln(p(z_{n + 1}|z_n)) + w_n(z_n) for every possible previous state z_n
                
                # find the max of those values, as well as the state z_n that maximized it
                max_val = max(dist.values()) 
                max_key = max(dist, key=dist.get)

                path_tracer[next_z] = max_key # save that maximized state
                next_w[next_z] = np.log(cond_prob) + max_val # w_{n + 1}(z_{n + 1}) = ln(p((x_{n + 1}, y_{n + 1})|z_{n + 1})) + max over z_n of ln(p(z_{n + 1}|z_n)) + w_n(z_n) 
            else:
                next_w[next_z] = np.log(cond_prob) # if the conditional probablity is 0, then we can skip the calculations above, and # w_{n + 1}(z_{n + 1}) = ln(p((x_{n + 1}, y_{n + 1})|z_{n + 1}))

        w[i + 1] = next_w # save the calculated w values
        max_states[i + 1] = path_tracer # save the previous states that maximized the corresponding w values

    # backwards pass
    estimated_hidden_states[num_time_steps - 1] = max(w[num_time_steps - 1], key=w[num_time_steps - 1].get) # the last estimated hidden state is the one that maximizes the last w value
    
    #  compute all the other estimated hidden states to find the sequence of max likelihood
    for i in reversed(range(num_time_steps-1)):
        estimated_hidden_states[i] = max_states[i + 1][estimated_hidden_states[i + 1]] # the most likely previous state is the one which is most likely to lead to the next state that we have already estimated

    return estimated_hidden_states


if __name__ == '__main__':
    
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                all_possible_observed_states,
                                prior_distribution,
                                rover.transition_model,
                                rover.observation_model,
                                observations)
    print('\n')


    
    timestep = num_time_steps - 1
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                                all_possible_observed_states,
                                prior_distribution,
                                rover.transition_model,
                                rover.observation_model,
                                observations)
    print('\n')
    
    timestep = num_time_steps - 1
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # computing the most likely state at each timestep from the forward-backward marginals
    print("Most likely state at each time using forward-backward algorithm: ")
    most_likely_states = [None]*num_time_steps
    for i in range(num_time_steps):
        likely_state = max(marginals[i], key=marginals[i].get) # most likely state at time i is the one that maximizes the marginal probabilities at time i
        most_likely_states[i] = likely_state
        print("z_{}: {}".format(i, likely_state))


    # Computing the error count of the viterbi sequence and the forward-backward sequence compared to the actual data
    f_b_error = 0
    viterbi_error = 0
    for i in range(num_time_steps):
        if most_likely_states[i] != hidden_states[i]:
            f_b_error += 1
        if estimated_states[i] != hidden_states[i]:
            viterbi_error += 1

    print("Forward-Backward Sequence Error Rate: {} \nViterbi Sequence Error Rate: {}".format(f_b_error/num_time_steps, viterbi_error/num_time_steps))

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                        observations,
                                        estimated_states,
                                        marginals)
        app.mainloop()
        
