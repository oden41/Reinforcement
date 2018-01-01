import gym
from gym import wrappers
import numpy as np
import time

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation, num_d):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_d)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_d)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_d)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_d))
    ]
    return sum([x * (num_d**i) for i, x in enumerate(digitized)])


def get_action(next_state, episode, q_table):
    # 徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q = max(q_table[next_state][0], q_table[next_state][1])
    q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                             alpha * (reward + gamma * next_Max_Q)

    return q_table


def main():
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './movie/cartpole-episode-1',force=True)
    #成功時の最大ステップ数
    max_number_of_steps = 200
    #成功基準の対象とする直近エピソード数
    num_consecutive_iterations = 100
    #最大エピソード数
    num_episodes = 2000
    #目標とする平均報酬
    goal_average_reward = 195
    #Q値テーブルを作成する際の離散化の個数
    num_dizitized = 6
    #Q値テーブル
    q_table = np.random.uniform(
        low=-1, high=1, size=(num_dizitized ** 4, env.action_space.n))

    total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
    final_x = np.zeros((num_episodes, 1))  # 学習後、各試行のt=200でのｘの位置を格納
    islearned = 0  # 学習が終わったフラグ
    isrender = 0  # 描画フラグ

    for episode in range(num_episodes):  # 試行数分繰り返す
        # 環境の初期化
        observation = env.reset()
        state = digitize_state(observation, num_dizitized)
        action = np.argmax(q_table[state])
        episode_reward = 0

        for t in range(max_number_of_steps):  # 1試行のループ
            if islearned == 1:  # 学習終了したらcartPoleを描画する
                env.render()
                time.sleep(0.1)
                print(observation[0])  # カートのx位置を出力

            # 行動a_tの実行により、s_{t+1}, r_{t}などを計算する
            observation, reward, done, info = env.step(action)

            # 報酬を設定し与える
            if done:
                if t < goal_average_reward:
                    reward = -200  # こけたら罰則
                else:
                    reward = 1  # 立ったまま終了時は罰則はなし
            else:
                reward = 1  # 各ステップで立ってたら報酬追加

            episode_reward += reward  # 報酬を追加

            # 離散状態s_{t+1}を求め、Q関数を更新する
            next_state = digitize_state(observation, num_dizitized)  # t+1での観測状態を、離散値に変換
            q_table = update_Qtable(q_table, state, action, reward, next_state)

            #  次の行動a_{t+1}を求める
            action = get_action(next_state, episode, q_table)  # a_{t+1}

            state = next_state

            # 終了時の処理
            if done:
                print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:],
                                              episode_reward))  # 報酬を記録
                if islearned == 1:  # 学習終わってたら最終のx座標を格納
                    final_x[episode, 0] = observation[0]
                break

        if (total_reward_vec.mean() >= goal_average_reward):  # 直近の100エピソードが規定報酬以上であれば成功
            print('Episode %d train agent successfuly!' % episode)
            islearned = 1
            np.savetxt('./learned_Q_table.csv',q_table, delimiter=",") #Qtableの保存する場合
            if isrender == 0:
                isrender = 1
        if islearned:
            np.savetxt('./final_x.csv', final_x, delimiter=",")
            env.close()
            return

if __name__ == '__main__':
    main()
