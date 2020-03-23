
from env import Board
from RL_brain import QLearningTable

size = 16
m = 8

def get_q_table(filename):
    q_table = {}
    f = open(filename, 'r+')
    lines = f.readlines()
    for line in lines:
        #line = line[:-1]
        state = line.split(';')[0]
        action_value = line.split(';')[1].split(',')
        q_table[state] = {}
        for i in range(len(action_value)-1):
            action = action_value[i].split(':')[0][1:]
            value = action_value[i].split(':')[1][1:]
            q_table[state][int(action)] = float(value)
    return q_table

def save_q_table(q_table,filename):
    f = open(filename, 'w+')
    f.write(str(q_table))
    f.close()


def update():
    for episode in range(100000000):
        # initial observation
        board.reset()
        state = board.get_state()

        if episode == 1000000:
            save_q_table(RL1.q_table)

        while True:
            # RL choose action based on observation
            action = RL1.choose_action(state)
            # RL take action and get next observation and reward
            reward, done, result = board.step(action, agent=False)
            state_ = board.get_state()
            # RL learn from this transition
            RL1.learn(state, action, reward, state_, done)
            if done:
                break

            # observation
            state = state_
            # RL choose action based on observation
            action = RL1.choose_action(state)
            # RL take action and get next observation and reward
            reward, done, result = board.step(action)
            state_ = board.get_state()
            # RL learn from this transition
            RL1.learn(state, action, reward, state_, done)
            # break while loop when end of this episode
            if done:
                break

            state = state_
            #count_space(state)

        print('episode is:',episode)

    # end of game
    print('training over')


if __name__ == "__main__":
    #input_file = 'Qtable6.txt'
    output_file = 'Qtable3.txt'

    #qtable = get_q_table(input_file)
    #print('get qtable!')
    board = Board(size, m)
    #RL1 = QLearningTable(size, q_table=qtable)
    RL1 = QLearningTable(size)

    update()
    #print(RL1.q_table)
    print(len(RL1.q_table))
    save_q_table(RL1.q_table, output_file)
    print('Finish')

