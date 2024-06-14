#### Step 1: Initialize the prior belief
import numpy as np
import matplotlib.pyplot as plt
import os

true_goal_proportion = 0.6
num_matches=10

goal_proportions = np.linspace(0, 1, 1000) # gives array

prior_belief=np.ones_like(goal_proportions)/len(goal_proportions) # gives array

#### Step 2: Likelihood Function


def likelihood(goal_proportion, outcome): # outcome is a bool, if goal is True, if not goal is false
    if outcome:
        #if goal is scored, the likelihood of observing this outcome is equal to goal_proportion
        return goal_proportion
    else:
        #if not, it means no goal is scored. 1- gives the percentage chance of observing no goal for the given goal_proportion
        return 1-goal_proportion

#### Step 3: Create the bayesian_update function

def bayesian_update(prior_belief, goal_proportions, outcome):
    likelihoods=np.array([likelihood(goal_proportion,outcome) for goal_proportion in goal_proportions])
    unnormalised_updated_b=prior_belief*likelihoods
    updated_b=unnormalised_updated_b/np.sum(unnormalised_updated_b)
    return updated_b


#### Step 4: Analyze football matches

#binomial distribution
#numpy.random.binomial(n, p, size=None), n number of trials, p propability of success in each trial, size number of matches, astype(bool) converts data into boolean
match_outcomes = np.random.binomial(1, true_goal_proportion, num_matches).astype(bool)
print("Match Outcomes:", match_outcomes)


#### Step 5: Update prior belief iteratively and store posterior distributions

posteriors = []

for i in range(1, len(match_outcomes) + 1):
    prior_belief = bayesian_update(prior_belief, goal_proportions, match_outcomes[i-1])
    posteriors.append(prior_belief)

    # Next Step: Visualize the analysis

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(goal_proportions, prior_belief, label='Initial Belief', linestyle='--')

for i, posterior in enumerate(posteriors):
    plt.plot(goal_proportions, posterior, label=f'Posterior after {i+1} match', alpha=0.7)

plt.xlabel('Proportion of Goals Scored')
plt.ylabel('Probability Density')
plt.title('Bayesian Analysis: Estimating Proportion of Goals Scored by a Football Team')
plt.legend()
plt.grid(True)

output_folder = os.path.join(os.path.dirname(__file__), '..', 'figures')
output_path = os.path.join(output_folder, 'bayesian_analysis_plot.png')

os.makedirs(output_folder, exist_ok=True)

plt.savefig(output_path)


plt.show()
