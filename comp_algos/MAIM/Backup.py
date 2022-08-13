import Environment as EN
import I_C
import time
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
Data = r"C:\Users\siwei\Desktop\Paper\Final Code\DATA\hep.txt"
testData = r"C:\Users\siwei\OneDrive\CNA\大创\complex_network_course-master\Homework_4\DBLP.txt"
max_reject = 5
Memory_size = 35

budget = 30
learning_frequency = 30
sigmoid = 0.7 #rewardEing factor

Round = 15
traing_turns = 5
DBLP_test_turns = 10

# [0, traing_turns) traing on Amazon
# [traing_turns,DBLP_test_turns) Performance on Amazon
# [DBLP_test_turns,Round]


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.03,
            reward_decay=0.9,
            e_greedy=0.8,
            replace_target_iter=20,
            memory_size=Memory_size,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.action_numbers = 0
        self.total_record_accept = list()
        self.total_record_reject = list()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            #fully connected layer with nfeatures input, 20 output and relu activation function.
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
            #try to give value function of all Q(s,a) by one set
            #In toy design, things will not go this way

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, feat_cur):
        # to have batch dimension when feed into tf placeholder
        action = 1
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            observation = np.mat(feat_cur)
            reject = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            self.total_record_accept += [reject[0][0]]
            self.total_record_reject += [reject[0][1]]
            print("Input: ",observation," output: ",reject, flush=True)
            action = np.argmax(reject)
        else:
            action = sample([0,1],1)[0]
        return action

    def choose_action_no_ran(self, feat_cur):
        # to have batch dimension when feed into tf placeholder
        action = 1
        observation = np.mat(feat_cur)
        reject = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        self.total_record_accept += [reject[0][0]]
        self.total_record_reject += [reject[0][1]]
        print("Input: ",observation," output: ",reject, flush=True)
        action = np.argmax(reject)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def reject_dis(self):
        return [a_i - b_i for a_i, b_i in zip(self.total_record_accept, self.total_record_reject)]

    def check_memory(self):
        for i in range(self.memory_size):
            print(self.memory[i,:],flush=True)
        pass

RL = DeepQNetwork(2,2, output_graph=True)

def run_it():
    step = 0
    action_times = 0 # times we have choose actions
    total_reward = list()
    CELF_reward = list()
    for episode in range(Round):
    # test times
        if episode == traing_turns:
            print("# -------------------WARNING: From training to Testing !---------------",flush=True)
        if episode == DBLP_test_turns:
            print("# -------------------WARNING: From Amazon to DBLP !---------------",flush=True)
        Inf = []
        Col = []
        marginalGain = []
        Address = Data
        if episode >= 30:
            Address =  testData
        test = EN.Env(Address,1)
        L = test.list # a sorted list of [node,influence]

        # current step and max step can take
        step_c = 0
        while(True):
            print(len(test.seed))
            if len(test.seed) == budget:
                break
            elif step_c + 100 > len(test.list):
                print("Error !!")
                break
            input_at_step_test  = []
            if episode < traing_turns:
                action = RL.choose_action(test.node2feat(step_c))
            else:
                action = RL.choose_action_no_ran(test.node2feat(step_c))
            print("Occupied budget: ",len(test.seed))
            # made a decision I think is right

            if action == 0: #accepted\
                print("Round: ",episode ,"Current node: ",step_c, " accepted")
                input_at_step_test = test.node2feat(step_c)
                reward = test.steps(step_c,1) - L[step_c + 1][1] * sigmoid
                step_next = step_c + 1
            else:
                print("Round: ",episode ,"Current node: ",step_c, " rejected")
                input_at_step_test = test.node2feat(step_c)
                reward = L[step_c + 1][1] * sigmoid - test.steps(step_c,0)
                step_next = step_c + 1

            RL.store_transition(input_at_step_test, action, reward, test.node2feat(step_next))

            action_times += 1
            if (action_times > Memory_size) and (action_times % learning_frequency == 0) and (episode<traing_turns):
                print("Learning ...............................",flush=True)
                RL.learn()
            step_c = step_next
            Inf += [input_at_step_test[0]]
            Col += [input_at_step_test[1]]
            marginalGain += [reward]

        #I_C.I_C_relation(Inf, Col, marginalGain)
        total_reward = total_reward + [EN.IC(test.graph,test.seed)]
        print("Current influence is: ",EN.IC(test.graph,test.seed),flush=True)
        del test

    # end of game
    #test = EN.Env(Data)
    #_, A = CELF.CELF_plus_plus(test.graph,budget)
    #print(A,flush=True)
    #print(CELF.CELF_plus_plus(test.graph, 100))
    RL.check_memory()
    plt.subplot(121)
    plt.plot(range(Round),total_reward)
    plt.subplot(122)
    tmp = RL.reject_dis()
    plt.plot(range(len(tmp)),tmp)
    plt.show()
    print('game over')

run_it()
