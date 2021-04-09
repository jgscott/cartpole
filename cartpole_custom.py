import gym
import math 
import numpy as np
from gym import spaces, logger
from gym.utils import seeding


# A simple cartpole agent that implements Q learning via a tabular method.
# In the constructor, cutpoints is a n_cuts x 4 array telling the agent how
# to discretize the continuous 4D state space.


class CartPoleTabularAgent():
    def __init__(self, env, cutpoints, epsilon = 0.1):
        # cutpoints: an n_buckets x 4 array of cutpoints for
        # the discrete state-space representation
        n_buckets = cutpoints.shape[0] + 1
        self.Q = np.zeros((n_buckets, n_buckets, n_buckets, n_buckets, 2))
        self.action_space = env.action_space
        self.cutpoints = cutpoints
        self.n_buckets = n_buckets
        
    def discretize_state(self, state_real):
        state_bucket = [np.digitize(state_real[i], self.cutpoints[:,i]) for i in range(4)]
        return tuple(state_bucket)
        
    def choose_action(self, state, eps = 0.05):
    # epsilon-greedy action choice
        if(random.random() < eps):
            action = self.action_space.sample()
        else:
            state_bucket = self.discretize_state(state)
            q_vec = self.Q[(*state_bucket),...]
            if(q_vec[0] == q_vec[1]):
                action = self.action_space.sample()
            else:
                action = np.argmax(self.Q[(*state_bucket),...])
        return action
    
    def show_n_episodes(self, env, n, verbose=True):
        for i in range(n):
            state = env.reset()
            done = False
            step_counter = 0
            sum_rewards = 0
            while not done and step_counter < 500:
                env.render(mode='human')
                action = self.choose_action(state, eps=0) # greedy policy
                state, reward, done, info = env.step(action)
                sum_rewards += reward
                step_counter += 1
            if verbose:
                print(step_counter)
        env.close()
    
class CartPoleSarsaLambdaLearner(CartPoleTabularAgent):
    
    def __init__(self, env, cutpoints, epsilon = 0.1):
        CartPoleAgent.__init__(self, env, cutpoints, epsilon)
        self.Z = np.zeros((self.n_buckets, self.n_buckets, self.n_buckets, self.n_buckets, 2))
        
    def resetZ(self):
        # reset the eligibility trace to 0
        self.Z[...] = 0.0
    
    def update_Q(self, SARSA, alpha=0.001, gamma=1.0, lam = 0.99):
        current_state, action, reward, next_state, next_action = SARSA
        
        # discretize
        current_state_bucket = self.discretize_state(current_state)
        next_state_bucket = self.discretize_state(next_state)
        
        # reward target
        current_Q = self.Q[(*current_state_bucket),action]
        next_Q = self.Q[(*next_state_bucket),next_action]
        delta = reward + gamma*next_Q - current_Q
        
        # update eligibility trace
        self.Z[(*current_state_bucket),action] += 1
        
        # update all states and decay eligibility trace
        self.Q += (alpha*delta)*self.Z
        self.Z = gamma*lam*self.Z
                
            
class CartPoleQLearner(CartPoleTabularAgent):   
    
    def __init__(self, env, cutpoints, epsilon = 0.1):
        CartPoleAgent.__init__(self, env, cutpoints, epsilon)
        
    def update_Q(self, step, alpha, gamma=1.0, method='Q'):
        state, action, reward, next_state = step
        state_bucket = self.discretize_state(state)
        next_state_bucket = self.discretize_state(next_state)
        current_Q = self.Q[(*state_bucket),action]
        best_Qp = np.amax(self.Q[(*next_state_bucket),...])  # Q-value of best action for successor state
        delta = reward + gamma*best_Qp - current_Q
        self.Q[(*state_bucket),action] = current_Q + alpha*delta
             

class CartPoleMCLearner(CartPoleTabularAgent):   
    
    def __init__(self, env, cutpoints, epsilon = 0.1):
        CartPoleAgent.__init__(self, env, cutpoints, epsilon)
        self.C = np.zeros((self.n_buckets, self.n_buckets, self.n_buckets, self.n_buckets, 2))
        
    def update_Q(self, episode, alpha=0.001, gamma=1.0, method='Q'):
        total_rewards = 0.0
        # Simple Monte Carlo control
        # work backwards to sum rewards and attribute those to each visited (state, action) pair in the episode
        for state, action, reward, next_state in reversed(episode):
            total_rewards = gamma*total_rewards + reward
            state_bucket = self.discretize_state(state)
            self.C[(*state_bucket),action] += 1
            current_Q = self.Q[(*state_bucket),action]
            current_C = self.C[(*state_bucket),action]
            step_size =  max(alpha, 1.0/current_C)
            self.Q[(*state_bucket),action] += step_size*(total_rewards - current_Q)

            


class CustomCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
        
        This is just cut and pasted from the OpenAI Gym CartPole-v1 environment,
        with a simple modification to "sparsify" the reward structure (see Reward
        section below).

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 0 for every step taken, -1 if it falls over

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 0.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
