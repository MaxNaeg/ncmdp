import tensorflow as tf

@tf.function
def calc_cumulative_return(returns:tf.constant, gamma:tf.constant, values)->tf.constant:
    """
    Calculate the cumulative return of the episode
    :param returns: The returns of the episode
    :param gamma: The discount factor
    :return: The cumulative return of the episode
    """

    discounts = gamma * tf.ones_like(returns)

    def discounted_return_fn(accumulated_discounted_reward, reward_discount):
        reward, discount = reward_discount
        return accumulated_discounted_reward * discount + reward
    
    return tf.scan(fn=discounted_return_fn, elems=(returns, discounts), initializer=tf.constant(0.0), reverse=True)



class REINFORCE_BASE():
    """Standard REINFORCE: train_critic=False, critic=None, critic_optimizer=None, 
    cum_rew_func doesn't use values, values are None."""

    def __init__(self, model, optimizer, critic=None, critic_optimizer=None, gamma=0.99, cum_rew_func=calc_cumulative_return, train_critic=True):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.cum_rew_func = cum_rew_func
        self.critic = critic
        self.train_critic = train_critic
        self.critic_optimizer = critic_optimizer

    def sample_action(self, observation, seed=42):
        logits= self.model(observation)
        action = tf.random.categorical(logits, 1, seed=seed)
        return action
    
    @tf.function
    def sample_value(self, observation):
        return self.critic(observation)
    
    def train_step(self, observations, actions, rewards, values):
        cum_rewards = self.cum_rew_func(rewards, self.gamma, values)
        if values is not None:
            cum_advantages = cum_rewards - tf.gather(values, tf.range(len(cum_rewards)))
        else:
            cum_advantages = cum_rewards
        self._train_actor(observations, actions, cum_advantages)
        if self.train_critic:
            self._train_critic(observations, cum_rewards)

    @tf.function
    def _train_critic(self, observations, cum_rewards):
        with tf.GradientTape() as tape:
            values = self.critic(observations)
            critic_loss = tf.reduce_mean(tf.square(tf.reshape(values, tf.constant([-1])) - cum_rewards))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    @tf.function
    def _train_actor(self, observations, actions, cum_rewards):
        tf.debugging.assert_all_finite(cum_rewards, 'Nan in cum_rewards')
        with tf.GradientTape() as tape:
            logits = self.model(observations)
            tf.debugging.assert_all_finite(logits, 'Nan in logits')
            norms = tf.math.log(tf.reduce_sum(tf.exp(logits), axis=-1))
            tf.debugging.assert_all_finite(norms, 'Nan in norms')
            policy_loss = -tf.reduce_mean(
                (tf.gather(logits, actions, axis=1, batch_dims=1) - norms) * cum_rewards)
            tf.debugging.assert_all_finite(policy_loss, 'Nan in policy loss')

        grads = tape.gradient(policy_loss, self.model.trainable_variables)
        tf.debugging.assert_all_finite(policy_loss, 'Nan in scaled grads')
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))