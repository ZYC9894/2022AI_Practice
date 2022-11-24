import numpy as np
import tensorflow as tf

from exp import AlgorithmBase


class Algorithm(AlgorithmBase):
    def __init__(self) -> None:
        super().__init__()

        self.is_reinforce_task_list = [
            True,
            True,
            True,
            True,
            True,
            True,
        ]  # means each task whether need reinforce
        self.batch_size = 512
        self.lstm_time_steps = 16

        self.label_size_list = [12, 16, 16, 16, 16, 8]
        self.seri_vec_split_shape = [(725,), (84,)]

        self.min_policy = 0.00001
        self.clip_param = 0.2
        self.var_beta = 0.025

        self.unsqueeze_label = None
        self.old_label_probability = None
        self.fc2_label = None
        self.unsqueeze_reward = None
        self.unsqueeze_advantage = None
        self.fc2_value_result = None
        self.seri_vec = None
        self.unsqueeze_weight_list = None

    def _squeeze_tensor(
        self,
        unsqueeze_reward,
        unsqueeze_advantage,
        unsqueeze_label_list,
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
        return reward, advantage, label_list, weight_list

    def _build_input_tensors(self):
        with tf.name_scope("Placeholder"):
            self.unsqueeze_label = [
                tf.placeholder(
                    tf.int32,
                    [self.batch_size * self.lstm_time_steps, 1],
                    name="label_%s" % i,
                )
                for i in range(len(self.label_size_list))
            ]
            self.old_label_probability = [
                tf.placeholder(
                    tf.float32,
                    [self.batch_size * self.lstm_time_steps, x],
                    name="old_label_probability_%s" % i,
                )
                for i, x in enumerate(self.label_size_list)
            ]

            self.fc2_label = [
                tf.placeholder(
                    tf.float32,
                    [self.batch_size * self.lstm_time_steps, x],
                    name="fc2_label_%s" % i,
                )
                for i, x in enumerate(self.label_size_list)
            ]

            self.unsqueeze_reward = tf.placeholder(
                tf.float32, [self.batch_size * self.lstm_time_steps, 1], name="reward"
            )
            self.unsqueeze_advantage = tf.placeholder(
                tf.float32,
                [self.batch_size * self.lstm_time_steps, 1],
                name="advantage",
            )

            self.fc2_value_result = tf.placeholder(
                tf.float32,
                [self.batch_size * self.lstm_time_steps, 1],
                name="fc2_value_result",
            )

            self.seri_vec = tf.placeholder(
                tf.float32,
                [
                    self.batch_size * self.lstm_time_steps,
                    np.sum(self.seri_vec_split_shape),
                ],
                name="seri_vec",
            )

            self.unsqueeze_weight_list = [
                tf.placeholder(
                    tf.float32,
                    [self.batch_size * self.lstm_time_steps, 1],
                    name="weight_list_%s" % i,
                )
                for i in range(len(self.label_size_list))
            ]
        self.input_tensors = [
            *self.unsqueeze_label,
            *self.old_label_probability,
            *self.fc2_label,
            self.unsqueeze_reward,
            self.unsqueeze_advantage,
            self.fc2_value_result,
            self.seri_vec,
            *self.unsqueeze_weight_list,
        ]
        tf.Variable(tf.constant("Hello World", name="hello"))

    def random_data(self):
        feed_dict = {}
        for i in range(len(self.label_size_list)):
            feed_dict["Placeholder/label_%s:0" % i] = np.random.randint(
                self.label_size_list[i],
                size=(self.batch_size * self.lstm_time_steps, 1),
            )
            feed_dict["Placeholder/old_label_probability_%s:0" % i] = np.random.rand(
                self.batch_size * self.lstm_time_steps, self.label_size_list[i]
            )
            feed_dict["Placeholder/fc2_label_%s:0" % i] = np.random.rand(
                self.batch_size * self.lstm_time_steps,
                self.label_size_list[i],
            )
            feed_dict["Placeholder/weight_list_%s:0" % i] = np.random.rand(
                self.batch_size * self.lstm_time_steps, 1
            )
        feed_dict["Placeholder/reward:0"] = np.random.rand(
            self.batch_size * self.lstm_time_steps, 1
        )
        feed_dict["Placeholder/advantage:0"] = np.random.rand(
            self.batch_size * self.lstm_time_steps, 1
        )
        feed_dict["Placeholder/fc2_value_result:0"] = np.random.rand(
            self.batch_size * self.lstm_time_steps, 1
        )
        feed_dict["Placeholder/seri_vec:0"] = np.random.rand(
            self.batch_size * self.lstm_time_steps, np.sum(self.seri_vec_split_shape)
        )
        return feed_dict

    def _build_output_tensor(self):
        unsqueeze_label_list = self.unsqueeze_label
        old_label_probability_list = self.old_label_probability
        fc2_label_list = self.fc2_label
        unsqueeze_reward = self.unsqueeze_reward
        unsqueeze_advantage = self.unsqueeze_advantage
        fc2_value_result = self.fc2_value_result
        seri_vec = self.seri_vec
        unsqueeze_weight_list = self.unsqueeze_weight_list

        reward, advantage, label_list, weight_list = self._squeeze_tensor(
            unsqueeze_reward,
            unsqueeze_advantage,
            unsqueeze_label_list,
            unsqueeze_weight_list,
        )
        _, split_feature_legal_action = tf.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            axis=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = tf.reshape(
            split_feature_legal_action, feature_legal_action_shape
        )

        legal_action_flag_list = tf.split(
            feature_legal_action, self.label_size_list, axis=1
        )

        # loss of value net
        fc2_value_result_squeezed = tf.squeeze(fc2_value_result, axis=[1])

        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * tf.reduce_mean(tf.square(new_advantage), axis=0)

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []
        # policy loss: ppo clip loss
        self.policy_cost = tf.constant(0.0, dtype=tf.float32)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = tf.constant(0.0, dtype=tf.float32)
                one_hot_actions = tf.one_hot(
                    label_list[task_index], self.label_size_list[task_index]
                )
                legal_action_flag_list_max_mask = (
                    1 - legal_action_flag_list[task_index]
                ) * tf.pow(10.0, 20.0)
                label_logits_subtract_max = tf.clip_by_value(
                    (
                        fc2_label_list[task_index]
                        - tf.reduce_max(
                            fc2_label_list[task_index]
                            - legal_action_flag_list_max_mask,
                            axis=1,
                            keep_dims=True,
                        )
                    ),
                    -tf.pow(10.0, 20.0),
                    1,
                )
                label_logits_subtract_max_list.append(label_logits_subtract_max)
                label_exp_logits = (
                    legal_action_flag_list[task_index]
                    * tf.exp(label_logits_subtract_max)
                    + self.min_policy
                )
                label_sum_exp_logits = tf.reduce_sum(
                    label_exp_logits, axis=1, keep_dims=True
                )
                label_sum_exp_logits_list.append(label_sum_exp_logits)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)
                policy_p = tf.reduce_sum(one_hot_actions * label_probability, axis=1)
                policy_log_p = tf.log(policy_p + 0.00001)
                old_policy_p = tf.reduce_sum(
                    one_hot_actions * old_label_probability_list[task_index] + 0.00001,
                    axis=1,
                )
                old_policy_log_p = tf.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = tf.exp(final_log_p)
                clip_ratio = tf.clip_by_value(ratio, 0.0, 3.0)
                surr1 = clip_ratio * advantage
                surr2 = (
                    tf.clip_by_value(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * advantage
                )
                temp_policy_loss = -tf.reduce_sum(
                    tf.to_float(weight_list[task_index]) * tf.minimum(surr1, surr2)
                ) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0)
                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -tf.reduce_sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * tf.log(
                        label_probability_list[current_entropy_loss_index] + 0.00001
                    ),
                    axis=1,
                )
                temp_entropy_loss = -tf.reduce_sum(
                    (temp_entropy_loss * tf.to_float(weight_list[task_index]))
                ) / tf.maximum(
                    tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0
                )  # add - because need to minize
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = tf.constant(0.0, dtype=tf.float32)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element
        self.entropy_cost_list = entropy_loss_list
        # sum all type cost
        self.cost_all = (
            self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        )
        return [self.cost_all]
