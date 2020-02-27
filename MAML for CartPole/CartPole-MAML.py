import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
tf.random.set_seed(0)
from tqdm import tqdm
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import random

def generateEnv():
    env = gym.make('CartPole-v1')
    env.gravity = np.random.normal(9.8, 1)
    env.masscart = np.random.normal(1.0,0.2)
    env.masspole = np.random.normal(0.1, 0.02)
    return env

class DQNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = keras.layers.Dense(24, input_shape=(4,), activation='relu')
        self.hidden2 = keras.layers.Dense(24, activation='relu')
        self.outputlayer = keras.layers.Dense(2, activation="linear")

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.outputlayer(x)
        return x

class DQNReplayer:
    def __init__(self):
        self.memory = pd.DataFrame(columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.count = 0
    
    def store(self, *args):
        self.memory.loc[self.count] = args
        self.count = self.count + 1
        
    def sample(self, size=None):
        if(size is None):
             return (np.stack(self.memory.loc[range(self.count), field]) for field in self.memory.columns)
        else:
            indices = np.random.choice(self.count, size=size)
            return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

def loss_function(pred_y, y):
  return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits

def copy_model(model, x):
    copied_model = DQNModel()
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

class MAMLDQNAgent():
    def __init__(self, alpha, epsilon, batch, K, gamma, maximumStep=2000):
        self.alpha = alpha
        self.model = DQNModel()
        self.epsilon = epsilon
        self.batch = batch
        self.K = K
        self.gamma = gamma
        self.maximumStep = maximumStep
        self.action_n = 2
        self.episodes_reward = []
        self.validation_model = DQNModel()

    def decide(self, observation, model):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = model.forward(observation[np.newaxis])
        return np.argmax(qs)

    def play_qlearning(self, env, model, replayer=None):
        observation = env.reset()
        model.forward(observation[np.newaxis])
        steps = 0
        episode_reward = 0
        if (replayer is None):
            replayer = DQNReplayer()
        while True:
            action = self.decide(observation,model)
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            replayer.store(observation, action, reward, next_observation, done)
            steps = steps + 1
            # if done or steps>=self.maximumStep:
            if done:
                self.episodes_reward.append(episode_reward)
                break
            observation = next_observation
        return replayer, episode_reward

    def sampleTrajectories(self, model, env):
        replayer = DQNReplayer()
        for _ in range(self.K):
            replayer, _ = self.play_qlearning(env, model, replayer)
        return replayer

    def sampleTasks(self, generateEnv):
        Env=[]
        for _ in range(self.batch):
            Env.append(generateEnv())
        return Env

    def generateTrainingData(self, replayer, model):
        observations, actions, rewards, next_observations, dones = replayer.sample()
        next_qs = model.forward(next_observations).numpy()
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. -  dones) * next_max_qs
        targets = model.forward(observations).numpy()
        targets[np.arange(us.shape[0]), actions] = us
        return observations, targets
    
    def trainOnce(self):
        optimizer = keras.optimizers.Adam()
        Env = self.sampleTasks(generateEnv)
        with tf.GradientTape() as test_tape:
            overallLoss = 0
            for i,env in enumerate(Env):
                replayer = self.sampleTrajectories(self.model, env)
                observations, targets = self.generateTrainingData(replayer, self.model)
                with tf.GradientTape() as train_tape:
                    train_loss, _ = compute_loss(self.model, observations, targets)
                gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
                k = 0
                model_copy = copy_model(self.model, observations)
                for j in range(len(model_copy.layers)):
                    model_copy.layers[j].kernel = tf.subtract(self.model.layers[j].kernel,
                                tf.multiply(self.alpha, gradients[k]))
                    model_copy.layers[j].bias = tf.subtract(self.model.layers[j].bias,
                                tf.multiply(self.alpha, gradients[k+1]))
                    k += 2
                replayer_test = self.sampleTrajectories(model_copy, env)
                observation_test, targets_test = self.generateTrainingData(replayer_test, model_copy)
                test_loss, _ = compute_loss(model_copy, observation_test, targets_test)
                overallLoss += test_loss
        gradients = test_tape.gradient(overallLoss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def validation(self, episodes, env, valid = True):
        optimizer = keras.optimizers.Adam()
        for i in range(episodes):
            replayer = self.sampleTrajectories(self.validation_model, env)
            observations, targets = self.generateTrainingData(replayer, self.validation_model)
            with tf.GradientTape() as valid_tape:
                train_loss, _ = compute_loss(self.validation_model, observations, targets)
            gradients = valid_tape.gradient(train_loss, self.validation_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.validation_model.trainable_variables))
        if(valid):
            epsilon = self.epsilon
            self.epsilon = 0
            _, episode_reward = self.play_qlearning(env, self.validation_model, replayer=None)
            self.epsilon = epsilon
            return episode_reward
        return 0
        # self.epsilon = epsilon

def MAMLTraining(filename = 'ex1.txt'):
    # f = open(filename, 'w')
    tf.keras.backend.set_floatx('float64')
    agent = MAMLDQNAgent(alpha=0.001, epsilon=0.5, batch=10, K=2, gamma=0.9)
    Average_reward=[]
    last_episode=[]
    for i in range(1500):
        agent.trainOnce()
        last_episode.append(agent.episodes_reward[-1])
        if(i%5 == 0):
            Average_reward.append(np.mean(agent.episodes_reward))
            print(i, '：Last Reward：', agent.episodes_reward[-1], 'Average Reward：', Average_reward[-1])
            # f.write(str(i)+ '：Last Reward：'+str(agent.episodes_reward[-1])+ 'Average Reward：'+str( np.mean(agent.episodes_reward)))
        # if(i%50==0):
            # agent.model.save_weights('MAML_weights/maml_car'+str(i)+'.h5')
# agent.model._set_inputs()

    np.save('variables/cartpole_MAML_training_episode_reward.npy', agent.episodes_reward)
    np.save('variables/cartpole_MAML_training_last_episode_reward.npy', last_episode)
    # f.close()
    agent.validation_model = copy_model(agent.model, np.zeros([1,4]))
    agent.validation(10, env= generateEnv())
    plt.figure()
    plt.plot(agent.episodes_reward)
    plt.show()
    # plt.savefig('ex1.png')

def validation():
    tf.keras.backend.set_floatx('float64')
    agent = MAMLDQNAgent(alpha=0.001, epsilon=0.3, batch=10, K=2, gamma=0.9)
    agent2 = MAMLDQNAgent(alpha=0.001, epsilon=0.3, batch=10, K=2, gamma=0.9)
    agent.validation_model.forward(np.array([0,0,0,0])[np.newaxis])
    agent2.validation_model.forward(np.array([0,0,0,0])[np.newaxis])
    averageRewardMAML = []
    averageReward= []
    for i in range(30):
        reward=[]
        reward2=[]
        for _ in range(10):
            agent.validation_model.load_weights('MAML_weights/maml_car'+str(i*50)+'.h5')
            agent2.validation_model.load_weights('Weights/maml_car'+str(i)+'.h5')
            reward.append(agent.validation(3, generateEnv()))
            reward2.append(agent2.validation(3, generateEnv()))
        averageRewardMAML.append(np.mean(reward))
        averageReward.append(np.mean(reward2))
        print(averageRewardMAML[-1],'; ', averageReward[-1])
    np.save('variables/cartpole_comparision_MAML_finetuning.npy', [averageRewardMAML, averageReward])
    plt.figure()
    plt.plot([i for i in range(30)], averageRewardMAML)
    plt.plot([i for i in range(30)], averageReward)
    plt.xlabel('Training episodes')
    plt.ylabel('Reward')
    plt.show()

# validation()

def Training():
    tf.keras.backend.set_floatx('float64')
    episode_rewards=[]
    agent = MAMLDQNAgent(alpha=0.001, epsilon=0.5, batch=10, K=2, gamma=0.9)
    agent.model.forward(np.array([0,0,0,0])[np.newaxis])
    for i in range(30):
        agent.validation(100, gym.make('CartPole-v1'), False)
        agent.validation_model.save_weights('Weights/maml_car'+str(i)+'.h5')
        print(i, ':', agent.episodes_reward[-1])
    plt.figure()
    plt.plot(agent.episodes_reward)
    plt.show()
    np.save('variables/cartpole_training_episode_reward.npy', agent.episodes_reward)

validation()