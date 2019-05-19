#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 310a5c89-3ea1-11e9-b0fd-00505601122b
# 90257956-3ea2-11e9-b0fd-00505601122b
# 69bef76d-1ebb-11e8-9de3-00505601122b
import numpy as np

import cart_pole_evaluator

#learning_rate = 0.1
discount_factor = 1.0

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2 * 1024, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(42)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=True)

    # Create Q, C and other variables
    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing estimated Q value of a given (state, action) pair.
    Q = np.zeros([env.states, env.actions])

    # - Create C, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing number of observed returns of a given (state, action) pair.
    C = np.zeros([env.states, env.actions])

    for _ in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards = [], [], []
        while True:
            epsilon = args.epsilon
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
                epsilon = 0

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random actions (there are env.actions of them),
            # otherwise, choose and action with maximum Q[state, action].
            action = np.random.randint(0, env.actions) if np.random.rand() < epsilon else np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # TODO: Compute returns from the observed rewards.
        returns = 0
        for i in reversed(range(len(states))):
            returns = rewards[i] + discount_factor * returns

            # TODO: Update Q and C
            C[states[i], actions[i]] += 1
            Q[states[i], actions[i]] = Q[states[i], actions[i]] + (1 / C[states[i], actions[i]]) * (returns - Q[states[i], actions[i]])

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
