import time 
import numpy as np 
from IPython.display import clear_output

import matplotlib.pyplot as plt 
import seaborn as sns 

from .viz import viz

################################################
#                                              #
#         THE FROZEN LAKE ENVIRONMENT          # 
#                                         @ZF  #
################################################

layout = [
    "S.......",
    "........",
    "...H....",
    ".....H..",
    "...H....",
    ".HH...H.",
    ".H..H.H.",
    "...H...G"
]

class frozen_lake:
    n_row = 8
    n_col = 8

    def __init__(self, layout=layout, eps=.2, seed=1234, Rscale=1):

        # get occupancy 
        self.rng = np.random.RandomState(seed)
        self.layout = layout
        self.get_occupancy()
        self.eps = eps 
        self.Rscale = Rscale
        # define MDP 
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()
        

    def get_occupancy(self):
        # get occupancy, current state and goal 
        map_dict = {
            'H': .7,
            '.': 0,
            'S': 0, 
            'G': 0,
        }
        self.occupancy = np.array([list(map(lambda x: map_dict[x], row)) 
                                   for row in self.layout])
        self.goal = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='G'))
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        holes = np.array([list(row) for row in self.layout])=='H'
        self.hole_cells = [h for h in np.vstack(np.where(holes)).T]
        
    def cell2state(self, cell):
        return cell[0]*self.occupancy.shape[1] + cell[1]
    
    def state2cell(self, state):
        n = self.occupancy.shape[1]
        return np.array([state//n, state%n])
        
    # ------------------ Define MDP --------------- #

    def _init_S(self):
        '''Define the state space
        '''
        self.nS = frozen_lake.n_row*frozen_lake.n_col
        self.S  = list(range(self.nS))
        self.goal_state = self.cell2state(self.goal)
        self.state = self.cell2state(self.curr_cell)
        self.hole_states = [self.cell2state(h) for h in self.hole_cells]
        self.s_termination = self.hole_states+[self.goal_state]

    def _init_A(self,):
        '''Define the action space 
        '''
        # init 
        self.directs = [
            np.array([-1, 0]), # up
            np.array([ 1, 0]), # down
            np.array([ 0,-1]), # left
            np.array([ 0, 1]), # right
        ]
        self.nA = len(self.directs)
        self.A  = list((range(self.nA)))

    def _init_P(self):
        '''Define the transition function, P(s'|s,a)

            P(s'|s,a) is a probability distribution
        '''

        def p_s_next(s, a):
            p_next = np.zeros([self.nS])
            cell = self.state2cell(s)
            # if the current state is terminal state
            # state in the current state 
            if s in self.s_termination:
                p_next[s] = 1 
            else:
                for j in self.A:
                    s_next = self.cell2state(
                        np.clip(cell + self.directs[j],
                        0, frozen_lake.n_row-1))
                    # the agent is walking on a surface of frozen ice, they cannot always
                    # successfully perform the intended action. For example, attempting to move "left"
                    # may result in the agent moving to the left with a probability of 1-ε.
                    # With probability ε, the agent will randomly move in one of the 
                    # other possible directions.
                    if j == a: 
                        p_next[s_next] += 1-self.eps
                    else:
                        p_next[s_next] += self.eps / (self.nA-1)
                
            return p_next
        
        self.p_s_next = p_s_next

    def _init_R(self):
        '''Define the reward function, R(s,a,s')

        return:
            r: reward
            done: if terminated 
        '''
        def R(s):
            if s == self.goal_state:
                return 1*self.Rscale, True
            elif s in self.hole_states:
                return -1, True
            else:
                return 0, False
        self.r = R
        
    def reset(self):
        '''Reset the environment

            Bring the agent back to the starting point
        '''
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        self.state = self.cell2state(self.curr_cell)
        self.done = False
        self.act = None

        return self.state, None, self.done 
    
    # ------------ visualize the environment ----------- #

    def render(self, ax, epi=False, step=False):
        '''Visualize the current environment
        '''
        occupancy = np.array(self.occupancy)
        sns.heatmap(occupancy, cmap=viz.mixMap, ax=ax,
                    vmin=0, vmax=1, 
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=occupancy.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=occupancy.shape[1], color='k',lw=5)
        ax.text(self.goal[1]+.15, self.goal[0]+.75, 'G', color=viz.Red,
                    fontweight='bold', fontsize=10)
        ax.text(self.curr_cell[1]+.25, self.curr_cell[0]+.75, 'O', color=viz.Red,
                    fontweight='bold', fontsize=10)
        r, _ = self.r(self.state)
        title = f'Episode: {epi}, Step: {step}' if epi else f'Reward: {r}, done: {self.done}' 
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect(1)

    def show_pi(self, ax, pi):
        '''Visualize your policy π(a|s)
        '''
        #self.reset()
        self.render(ax)
        for s in self.S:
            if s not in self.s_termination:
                cell = self.state2cell(s)
                for a in self.A:
                    pa = pi[s, a]
                    if pa > 0:
                        next_cell = self.directs[a]*.25
                        ax.arrow(cell[1]+.5, cell[0]+.5, 
                                next_cell[1]*pa, next_cell[0]*pa,
                                width=.005, color='k')
        ax.set_title('Policy')

    def show_v(self, ax, V):
        '''Visualize the value V(s) for each state given a policy
        '''
        v_mat = V.reshape([frozen_lake.n_row, frozen_lake.n_col])
        sns.heatmap(v_mat, cmap=viz.RedsMap, ax=ax,
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=v_mat.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=v_mat.shape[1], color='k',lw=5)
        for s in self.S:
            if s not in self.s_termination:
                    cell = self.state2cell(s)
                    v = V[s].round(2)
                    ax.text(cell[1]+.15, cell[0]+.65,
                            str(v), color='k',
                            fontweight='bold', fontsize=8)
        ax.set_title('Value')
        ax.set_axis_off()
        ax.set_box_aspect(1)

    # ------------ interact with the environment ----------- #
    
    def step(self, act):
        '''Update the state of the environment
        '''
        p_s_next = self.p_s_next(self.state, act)
        self.state = self.rng.choice(self.S, p=p_s_next)
        self.curr_cell = self.state2cell(self.state)
        rew, self.done = self.r(self.state)
        self.act = None 
        return self.state, rew, self.done

class cart_pole:

    def __init__(self, seed=1234, dt=0.02, gravity=9.8, mass_cart=1.0, mass_pole=0.1, length=0.7, force_mag=10.0, tau=0.02):
        np.random.seed(seed)
        # Physical constants
        self.dt = dt                # time step (s)
        self.gravity = gravity      # gravitational constant
        self.mass_cart = mass_cart  # mass of the cart (kg)
        self.mass_pole = mass_pole  # mass of the pole (kg)
        self.length = length        # length of the pole (m)
        self.force_mag = force_mag  # maximum magnitude of the force (N)
        self.tau = tau  # time interval for the simulation (s)

        # State variables
        self.state = np.zeros(4)  # [x, x_dot, theta, theta_dot]
        self.nS = 4
        
        # The action space is continuous: force magnitude between -1 and 1
        # We will apply the action to force the cart to move
        self.action_space = (-1.0, 1.0)
        
    def reset(self):
        # Reset the state to a random initial position and velocity
        self.state = np.random.uniform(low=-0.05, high=0.05, size=4)  # Random initial state
        return np.copy(self.state), 0, False
    
    def step(self, action):
        # Extract state variables
        x, x_dot, theta, theta_dot = self.state
        
        # Apply the continuous action (force between -1 and 1)
        force = action * self.force_mag
        
        # Physics equations of motion
        # Based on the linearized equations of the cart-pole system
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        total_mass = self.mass_cart + self.mass_pole
        pole_mass_length = self.mass_pole * self.length
        
        # Equations of motion
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (self.length * (4/3 - self.mass_pole * cos_theta**2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        # Update the state using Euler's method
        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc
        
        # Update the state
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check if the pole has fallen
        done = bool(theta < -np.pi/2 or theta > np.pi/2 or x < -2.4 or x > 2.4)
        
        # Reward: 1 for each time step the pole is balanced
        reward = 1.0
        
        return np.copy(self.state), reward, done
    
    def render(self, ax):
        # Visualization of the cart-pole system
        x, _, theta, _ = self.state
        cart_x = x
        pole_length = self.length
        pole_x = cart_x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
                
        # Draw the cart
        cart_width = 0.3
        cart_height = 0.24
        ax.add_patch(plt.Rectangle((cart_x - cart_width/2, -cart_height/2), cart_width, cart_height, color='blue'))
        
        # Draw the horizontal line
        ax.axhline(y=0, color='k',lw=3)

        # Draw the pole
        ax.plot([cart_x, pole_x], [0, pole_y], color='red', lw=3)
        ax.plot(cart_x, 0, 'ko')  # Draw the cart axle
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.3, 1.3)

    def close(self):
        plt.close()


if __name__ == '__main__':

    env = frozen_lake()
    env.reset()
    env.step(2)

    print(1)
        