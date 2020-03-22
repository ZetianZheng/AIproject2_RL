import sys
from env import Board
from RL_brain import QLearningTable

size = 16
m = 8

def get_q_table(filename):
    q_table = {}
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        #line = line[:-1]
        state = line.split(';')[0]
        action_value = line.split(';')[1].split(',')
        q_table[state] = {}
        for i in range(len(action_value)-1):
            action = action_value[i].split(':')[0][1]
            value = action_value[i].split(':')[1][1:]
            q_table[state][int(action)] = float(value)
    return q_table

def save_q_table(q_table,filename):
    f = open(filename, 'w+')
    for state in q_table:
        line = ''
        line += state + '; '
        for action in q_table[state]:
            value = q_table[state][action]
            line += str(action) + ': ' + str(value) + ', '
        f.write(line + '\n')


def update():
    for episode in range(int(sys.argv[1])):
        # initial observation
        board.reset()
        state = board.get_state()

        while True:
            # board.adversary_move()

            # RL choose action based on observation
            action = RL1.choose_action(state)

            # RL take action and get next observation and reward
            reward, done, result = board.step(action, agent=False)
            state_ = board.get_state()
            # RL learn from this transition
            #RL2.learn(state, action, reward, state_)
            RL1.learn(state, action, reward, state_, done)
            if done:
                break
            state = state_

            # observation

            # RL choose action based on observation
            action = RL1.choose_action(state)

            # RL take action and get next observation and reward
            reward, done, result = board.step(action)
            state_ = board.get_state()

            # RL learn from this transition
            #RL1.learn(state, action, reward, state_)
            RL1.learn(state, action, reward, state_, done)

            # break while loop when end of this episode
            if done:
                break
            state = state_

        print('episode is:',episode)

    # end of game
    print('train over')


if __name__ == "__main__":
    #input_file = sys.argv[2]
    output_file = sys.argv[2]

    board = Board(size, m)
    #qtable = get_q_table(input_file)
    RL1 = QLearningTable(size)
    #RL2 = QLearningTable(size)
    update()
    # print(RL1.q_table)
    print(len(RL1.q_table))
    save_q_table(RL1.q_table, output_file)
    #save_q_table(RL2.q_table,'Qtable2')
    #env.after(100, update)
    #env.mainloop()
