import gym
from gympackage.environment import Environment
from gympackage.agent import Agent

env = Environment('MountainCar-v0')
agent = Agent(env.num_action_space)
max_episode = 1000
env.initialize()
isdebug = True

for episode in range(max_episode):
    state = env.reset()
    for time in range(env.max_steps_episode):
        action = agent.get_action(state)
        obs = env.next_step(action)
        # 目標に達しない場合は報酬減
        if obs.done and time < env.reward_threshold:
            obs.reward = obs.reward - 200

        next_action = agent.get_action(obs.state)
        agent.update(state, action, obs.state, next_action)

        state = obs.state
        action = next_action

        if obs.done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, time + 1, env.get_reward_ave()))
            env.end()
            break

        if env.is_clear():
            print('Episode %d train agent successfuly!' % episode)
            exit()
