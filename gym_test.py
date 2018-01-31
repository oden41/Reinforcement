import gym

env = gym.make('CartPole-v0')
print(env)
print(env.action_space)
print(env.observation_space)
print(env.reward_range)
print(env.spec.max_episode_steps)#1エピソードの最大ステップ
print(env.spec.timestep_limit)#タイムステップの最大
print(env.spec.trials)
print(env.spec.reward_threshold)