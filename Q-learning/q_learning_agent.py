import pickle
import os
import numpy as np
import random
from environment import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 采样 <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        """
        使用Q-学习算法更新Q表格。

        参数:
        - state: 当前状态，用于查找当前状态对应的动作-值。
        - action: 当前采取的动作，用于查找当前状态对应的动作-值。
        - reward: 从当前状态采取动作后获得的即时奖励。
        - next_state: 从当前状态采取动作后进入的新状态，用于计算未来最大奖励。

        返回值:
        无直接返回值，此函数通过更新Q表格来改变对象状态。
        """
        # 获取当前状态-动作对应的动作值
        current_q = self.q_table[state][action]
        # 根据贝尔曼方程更新Q值
        # 通过当前奖励和未来可能的最大奖励计算新的Q值
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        # 更新Q表格，使当前状态-动作的Q值更加接近真实值
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # 从Q-table中选取动作
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(self.actions)
        else:
            # 从q表中选择
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    success_lst = []

    # 如果q表文件存在，则加载q表
    if os.path.exists('./q_table.pkl'):
        with open('./q_table.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
    for episode in range(100):
        print('episode: ', episode)
        state = env.reset()
        while True:
            env.render()
            # agent产生动作
            action = agent.get_action(str(state))
            next_state, reward, done, successful = env.step(action)
            # 走一步之后，才能开始学习
            # 更新Q表
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

            # 在单元格上输出Q表
            env.print_value_all(agent.q_table)
            # 当到达终点就终止游戏开始新一轮训练
            if done:
                success_lst.append(successful)
                break

    # 保存q表
    with open('./q_table.pkl', 'wb') as f:
        pickle.dump(dict(agent.q_table), f)

    # 绘制成功率曲线
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(0, len(success_lst))
    y = np.array(success_lst).cumsum()
    plt.plot(x, y)
    plt.savefig('./success_rate.png')
