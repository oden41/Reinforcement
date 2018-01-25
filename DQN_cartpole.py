import gym
from gym import wrappers
import numpy as np
import time
import random
from keras import backend as k
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
import tensorflow as tf


def huber_loss(y, y_pred, delta=1.0):
    err = y - y_pred
    cond = k.abs(err) < delta
    loss = tf.where(cond, 0.5 * k.square(err), delta * (k.abs(err) - 0.5 * delta))
    return k.mean(loss)


class Memory:   # stored as ( s, a, r, s_ )

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def is_full(self):
        return len(self.samples) >= self.capacity


class Q_NN:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huber_loss, optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, network):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                next_q = self.model.predict(next_state_b)[0]
                next_action = np.argmax(next_q)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * network.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号
            self.model.fit(inputs, targets, epochs=1, verbose=0)


class Agent:
    def get_action(self, state, episode, network):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0 + episode)

        if epsilon <= np.random.uniform(0, 1):
            qs = network.model.predict(state)[0]
            action = np.argmax(qs)
        else:
            action = np.random.choice([0, 1])  # ランダムに行動する

        return action

def main():
    env = gym.make('CartPole-v0')
    num_episodes = 1000  # 総試行回数
    max_number_of_steps = 200  # 1試行のstep数
    goal_average_reward = 195  # この報酬を超えると学習終了
    reward_length = 10  # 学習完了評価の平均計算を行う試行回数
    total_reward_vec = np.zeros(reward_length)  # 報酬ベクトル
    gamma = 0.99  # 割引係数
    islearned = 0  # 学習が終わったフラグ
    isrender = 0  # 描画フラグ

    hidden_size = 16  # Q-networkの隠れ層のニューロンの数
    learning_rate = 0.00001  # Q-networkの学習係数
    memory_size = 10000  # バッファーメモリの大きさ
    batch_size = 32  # Q-networkを更新するバッチサイズ

    mainQN = Q_NN(hidden_size=hidden_size, learning_rate=learning_rate)  # メインのQネットワーク
    targetQN = Q_NN(hidden_size=hidden_size, learning_rate=learning_rate)  # 価値を計算するQネットワーク
    plot_model(mainQN.model, to_file='./Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
    memory = Memory(memory_size)
    actor = Agent()

    for episode in range(num_episodes):  # 試行数分繰り返す
        env.reset()  # cartPoleの環境初期化
        state, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
        state = np.reshape(state, [1, 4])  # list型のstateを、1行4列の行列に変換
        episode_reward = 0

        targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする

        for t in range(max_number_of_steps + 1):  # 1試行のループ
            if islearned == 1:  # 学習終了したらcartPoleを描画する
                env.render()
                time.sleep(0.1)
                print(state[0, 0])  # カートのx位置を出力するならコメントはずす

            action = actor.get_action(state, episode, mainQN)  # 時刻tでの行動を決定する
            next_state, reward, done, info = env.step(action)  # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            next_state = np.reshape(next_state, [1, 4])  # list型のstateを、1行4列の行列に変換

            # 報酬を設定し、与える
            if done:
                next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
                if t < 195:
                    reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
                else:
                    reward = 1  # 立ったまま195step超えて終了時は報酬
            else:
                reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）

            episode_reward += 1  # reward  # 合計報酬を更新

            memory.add((state, action, reward, next_state))  # メモリの更新する
            state = next_state  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            mainQN.replay(memory, batch_size, gamma, targetQN)

            targetQN = mainQN

            # 1施行終了時の処理
            if done:
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                break

        # 複数施行の平均報酬で終了を判断
        if total_reward_vec.mean() >= goal_average_reward:
            print('Episode %d train agent successfully!' % episode)
            islearned = 1
            if isrender == 0:  # 学習済みフラグを更新
                isrender = 1
                env = gym.wrappers.Monitor(env, './movie/cartpoleDQN')


if __name__ == '__main__':
        main()
