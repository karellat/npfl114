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
import time
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Define suitable model, similarly to `reinforce` or `reinforce_with_baseline`.
        #
        # Use Adam optimizer with given `args.learning_rate`.

        input = tf.keras.layers.Input(shape=(80,80,3))
        conv = tf.keras.layers.MaxPool2D(4, 2)(input)
        conv = tf.keras.layers.Conv2D(16,3,2, padding='same')(conv)
        conv = tf.keras.layers.MaxPool2D(4, 2)(conv)
        conv = tf.keras.layers.Dropout(0.7)(conv)
        flatten = tf.keras.layers.Flatten()(conv)

        dense1 = tf.keras.layers.Dense(46, activation='relu')(flatten)
        output1 = tf.keras.layers.Dense(env.actions, activation='softmax')(dense1)

        self.model = tf.keras.Model(inputs=[input], outputs=[output1])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy()
            ]
        )

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states), np.array(actions), np.array(returns)

        # TODO: Train the model using the states, actions and observed returns.
        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        self.model.train_on_batch(states, actions, sample_weight=(returns))
        # - train the `baseline` model to predict `returns`

    def predict(self, states):
        states = np.array(states)

        # TODO: Predict distribution over actions for the given input states. Return
        # only the probabilities (if using a baseline).
        return self.model.predict_on_batch(states)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=3, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seed
    seed = 11111
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(env, args)
    
    # Training
    for _ in range(args.episodes // args.batch_size):
        if len(env._episode_returns) > 0 and np.mean(env._episode_returns[-100:]) >= 250:
            break

        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict([state])[0]
                # TODO(reinforce): Compute `action` according to the distribution returned by the network.
                # The `np.random.choice` method comes handy.
                action = np.random.choice(env.actions, p=probabilities)

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute `returns` from the observed `rewards`.
            returns = [0]
            for reward in reversed(rewards):
                returns.append(reward + returns[-1])
            returns = list(reversed(returns))
            returns.pop()

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)