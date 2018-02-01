import gym
from gympackage.environment import Environment

env = gym.make('CartPole-v1')
print(env)
print(env.action_space)
print(env.observation_space)
print(env.reward_range)
print(env.spec.max_episode_steps)#1エピソードの最大ステップ
print(env.spec.timestep_limit)#タイムステップの最大
print(env.spec.trials) #終了条件対象のエピソード数
print(env.spec.reward_threshold)
obs = env.reset()
e = Environment()
e.initialize()
obs = e.next_step(1)
print(obs)
