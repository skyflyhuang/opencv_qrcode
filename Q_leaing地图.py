import numpy as np
import matplotlib.pyplot as plt
# 定义迷宫地图
# 0表示可通行的空格，1表示墙壁，2表示目标点
maze = np.array([
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 2],
    [0, 0, 0, 1]
])

# 定义Q表，初始化为全0
Q_table = np.zeros((maze.shape[0], maze.shape[1], 4))

# 定义动作空间，分别表示上、下、左、右
actions = ['up', 'down', 'left', 'right']

# 定义参数
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000


# 定义获取下一个状态的函数
def get_next_state(state, action):
    if action == 'up':
        return max(0, state[0] - 1), state[1]
    elif action == 'down':
        return min(maze.shape[0] - 1, state[0] + 1), state[1]
    elif action == 'left':
        return state[0], max(0, state[1] - 1)
    elif action == 'right':
        return state[0], min(maze.shape[1] - 1, state[1] + 1)


# 强化学习算法
for episode in range(num_episodes):
    state = (0, 0)  # 初始状态
    while maze[state] != 2:  # 直到到达目标点
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q_table[state])]

        next_state = get_next_state(state, action)
        reward = -1 if maze[next_state] != 2 else 0

        # 更新Q值
        max_next_Q = np.max(Q_table[next_state])
        Q_table[state][actions.index(action)] += learning_rate * (
                    reward + discount_factor * max_next_Q - Q_table[state][actions.index(action)])

        state = next_state

# 输出最优策略
policy = np.zeros_like(maze)
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        policy[i][j] = np.argmax(Q_table[i][j])

# 可视化迷宫地图
plt.imshow(maze, cmap='binary')
plt.title('Maze')
plt.show()

print("Optimal Policy:")
# 可视化最优策略
actions_map = {0: '^', 1: 'v', 2: '<', 3: '>'}  # 对应动作的符号
policy_symbols = [[actions_map[int(policy[i][j])] for j in range(maze.shape[1])] for i in range(maze.shape[0])]
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        if maze[i][j] == 2:  # 目标点
            policy_symbols[i][j] = 'G'
        elif maze[i][j] == 1:  # 墙壁
            policy_symbols[i][j] = 'X'

plt.imshow(maze, cmap='binary')
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        plt.text(j, i, policy_symbols[i][j], ha='center', va='center')
plt.title('Optimal Policy')
plt.show()
print(policy)
