import numpy as np
import tensorflow as tf


from exp import AlgorithmBase


class Algorithm(AlgorithmBase):
    def __init__(self):
        super().__init__()
        # main camp soldier
        self.dim_of_soldier_1_10 = [18, 18, 18, 18]
        # enemy camp soldier
        self.dim_of_soldier_12_20 = [18, 18, 18, 18]
        # main camp organ
        self.dim_of_organ_1_2 = [18, 18]
        # enemy camp organ
        self.dim_of_organ_3_4 = [18, 18]
        # main camp hero
        self.dim_of_hero_frd = [235]
        # enemy camp hero
        self.dim_of_hero_emy = [235]
        # public hero info
        self.dim_of_hero_main = [14]  # main_hero_vec

        self.dim_of_global_info = [25]
        self.data_split_shape = [
            809,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            12,
            16,
            16,
            16,
            16,
            8,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            512,
            512,
        ]
        # self.batch_size = 512 * 16
        # self.lstm_time_steps = 16
        self.batch_size = 1
        self.lstm_time_steps = 1

        self.lstm_unit_size = 16
        self.seri_vec_split_shape = [(725,), (84,)]
        self.m_learning_rate = 0.0001
        self.m_var_beta = 0.025
        self.log_epsilon = 1e-6
        self.label_size_list = [12, 16, 16, 16, 16, 8]
        # self.need_reinforce_param_button_label_list = Config.NEED_REINFORCE_PARAM_BUTTON_LABEL_LIST
        self.is_reinforce_task_list = [
            True,
            True,
            True,
            True,
            True,
            True,
        ]  # means each task whether need reinforce
        self.min_policy = 0.00001
        self.clip_param = 0.2
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = 32
        self.data_shapes = [
            [12944],
            [16],
            [16],
            [16],
            [16],
            [16],
            [16],
            [16],
            [16],
            [192],
            [256],
            [256],
            [256],
            [256],
            [128],
            [16],
            [16],
            [16],
            [16],
            [16],
            [16],
            [16],
            [512],
            [512],
        ]
        self.cut_points = [value[0] for value in self.data_shapes]
        self.feature_dim = self.seri_vec_split_shape[0][0]
        self.legal_action_size_list = self.label_size_list.copy()
        self.legal_action_size_list[-1] = (
            self.legal_action_size_list[-1] * self.legal_action_size_list[0]
        )
        self.legal_action_dim = np.sum(self.legal_action_size_list)
        self.lstm_hidden_dim = self.lstm_unit_size
        self.legal_action_size = self.legal_action_size_list

    def _build_input_tensors(self):
        with tf.name_scope("Placeholder"):
            self.feature_ph = tf.placeholder(
                shape=(self.batch_size, self.feature_dim),
                name="feature",
                dtype=np.float32,
            )
            self.lstm_cell_ph = tf.placeholder(
                shape=(self.batch_size, self.lstm_hidden_dim),
                name="lstm_cell",
                dtype=np.float32,
            )
            self.lstm_hidden_ph = tf.placeholder(
                shape=(self.batch_size, self.lstm_hidden_dim),
                name="lstm_hidden",
                dtype=np.float32,
            )

        self.input_tensors = [
            self.feature_ph,
            self.lstm_cell_ph,
            self.lstm_hidden_ph,
        ]

    def _squeeze_tensor(
        self,
        unsqueeze_reward,
        unsqueeze_advantage,
        unsqueeze_label_list,
        unsqueeze_frame_is_train,
        unsqueeze_weight_list,
    ):
        reward = tf.squeeze(unsqueeze_reward, axis=[1])
        advantage = tf.squeeze(unsqueeze_advantage, axis=[1])
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(tf.squeeze(ele, axis=[1]))
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(tf.squeeze(weight, axis=[1]))
        frame_is_train = tf.squeeze(unsqueeze_frame_is_train, axis=[1])
        return reward, advantage, label_list, frame_is_train, weight_list

    def _fc_weight_variable(self, shape, name, trainable=True):
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )

    def _bias_variable(self, shape, name, trainable=True):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(
            name, shape=shape, initializer=initializer, trainable=trainable
        )

    def _build_output_tensor(self):
        split_feature_vec = tf.reshape(
            self.feature_ph, [-1, self.seri_vec_split_shape[0][0]]
        )
        init_lstm_cell, init_lstm_hidden = self.lstm_cell_ph, self.lstm_hidden_ph

        feature_vec_shape = list(self.seri_vec_split_shape[0])
        feature_vec_shape.insert(0, self.batch_size)
        feature_vec = tf.reshape(split_feature_vec, feature_vec_shape)
        feature_vec = tf.identity(feature_vec, name="feature_vec")

        lstm_cell_state = tf.reshape(init_lstm_cell, [-1, self.lstm_unit_size])
        lstm_hidden_state = tf.reshape(init_lstm_hidden, [-1, self.lstm_unit_size])
        lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
            lstm_cell_state, lstm_hidden_state
        )

        result_list = []

        hero_dim = (
            int(np.sum(self.dim_of_hero_frd))
            + int(np.sum(self.dim_of_hero_emy))
            + int(np.sum(self.dim_of_hero_main))
        )
        soldier_dim = int(np.sum(self.dim_of_soldier_1_10)) + int(
            np.sum(self.dim_of_soldier_12_20)
        )
        organ_dim = int(np.sum(self.dim_of_organ_1_2)) + int(
            np.sum(self.dim_of_organ_3_4)
        )
        global_info_dim = int(np.sum(self.dim_of_global_info))

        with tf.variable_scope("feature_vec_split"):
            feature_vec_split_list = tf.split(
                feature_vec, [hero_dim, soldier_dim, organ_dim, global_info_dim], axis=1
            )
            hero_vec_list = tf.split(
                feature_vec_split_list[0],
                [
                    int(np.sum(self.dim_of_hero_frd)),
                    int(np.sum(self.dim_of_hero_emy)),
                    int(np.sum(self.dim_of_hero_main)),
                ],
                axis=1,
            )
            soldier_vec_list = tf.split(
                feature_vec_split_list[1],
                [
                    int(np.sum(self.dim_of_soldier_1_10)),
                    int(np.sum(self.dim_of_soldier_12_20)),
                ],
                axis=1,
            )
            organ_vec_list = tf.split(
                feature_vec_split_list[2],
                [
                    int(np.sum(self.dim_of_organ_1_2)),
                    int(np.sum(self.dim_of_organ_3_4)),
                ],
                axis=1,
            )
            global_info_list = feature_vec_split_list[3]

            soldier_1_10 = tf.split(
                soldier_vec_list[0], self.dim_of_soldier_1_10, axis=1
            )
            soldier_11_20 = tf.split(
                soldier_vec_list[1], self.dim_of_soldier_12_20, axis=1
            )
            organ_1_2 = tf.split(organ_vec_list[0], self.dim_of_organ_1_2, axis=1)
            organ_3_4 = tf.split(organ_vec_list[1], self.dim_of_organ_3_4, axis=1)
            hero_frd = tf.split(hero_vec_list[0], self.dim_of_hero_frd, axis=1)
            hero_emy = tf.split(hero_vec_list[1], self.dim_of_hero_emy, axis=1)
            hero_main = tf.split(hero_vec_list[2], self.dim_of_hero_main, axis=1)
            global_info = global_info_list

        # TODO implement the output_tensors
        output_tensors = [
            tf.constant(0, shape=(1, 84), dtype=tf.float32),
            tf.constant(0, shape=(1, 1), dtype=tf.float32),
            tf.constant(0, shape=(1, 16), dtype=tf.float32),
            tf.constant(0, shape=(1, 16), dtype=tf.float32),
        ]
        return output_tensors
