import gym

env = gym.make('CentralDecision-v0')
for episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        adv_action = env.adv_action_space.sample()
        action.append(adv_action)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} time steps'.format(t + 1))
            break
env.close()