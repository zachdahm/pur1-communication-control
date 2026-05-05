'''
This is the control loop used for the paper: Model Predictive Control
 with State Estimation and Reduced-Order Physics for Nuclear Reactor
  Operation During Unanticipated Communication Transients."

Code was written by Zachery Dahm at Purdue University
'''


# Importing Packages
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import os
from gekko import GEKKO
import time
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
import warnings
import logging

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
##### Global Variables #####
INPUT_TIME = 40  # Input window length
OUTPUT_TIME = 10  # Output window length / step size
STEP_SIZE = 2 #This is listed as 2, but there is one second of overlap in how the data is handled so the control frequency is still 1hz
MAX_POWER = 3.815E7  # Maximum reactor power for normalization purposes
starting_baseline = 5712
baseline_reactivity = starting_baseline  # Excess reactivity (should be adjusted regularly)
counts_to_power_ratio = 4.379E9  # Converstion ratio (counts/power)
AMDR_START_POS = 30
AMDR_SPEED = 5
RR_SPEED = 55 / 60
SS_SPEED = 11 / 60

# Updated values
BETAS = np.array([0.00028, 0.00122, 0.00161, 0.00330, 0.00138, 0.00056])
LAMBDAS = np.array([0.01334, 0.03273, 0.12080, 0.456, 0.8502, 2.85582]) * 0.7

BETA = BETAS.sum()
L1 = BETA / (np.sum(BETAS / LAMBDAS))
LAMBDA = 7.358e-5  # prompt‐neutron lifetime [s]
SOURCE = 0.0  # external neutron source [n/s]
CHANNEL_1_TO_2 = 0.0000023357


@dataclass
#Dataclass for importing settings from the sensitivity analysis script
class SimConfig:
    # Controller
    method: str = 'MPC'  # 'MPC', 'PID',

    # Demand scenario (manages the shape of the demand curve)
    demand_scenario: str = '70% Steady State with ramp'

    #Duration of the trial
    num_iter: int = 1500

    # PID Terms (without feedforward)
    kp: float = 1.5
    ki: float = 0.005
    kd: float = 0.90

    #Feedforward PID terms
    #kp: float = 1.2
    #ki: float = 0.004
    #kd: float = 0.0
    kff: float = 0.5
    feedforward: bool = False

    # Communication / packet loss
    scenario: str = 'DoS'  # 'Nominal', 'DoS', 'Stochastic Packet Loss'
    burst_len: int = 600  # mean blackout duration in simulation steps
    burst_prob: float = 0.005 # probability per step of entering a burst (set to 0.005 for the paper's DoS probability)
    stochastic_threshold: float = 0.01 #per-second probability of packet loss, 0.01 is nominal, 0.90 is high-instensity

    # Kinetic model parameters (MPC internal model)
    beta_scale: float = 1.01  # multiplicative perturbation on all BETAS
    lambda_scale: float = 1.01  # multiplicative perturbation on LAMBDA

    # MHE tuning
    mhe_window: int = 10  # estimator horizon in steps

    # Rod position rounding
    discretization: float = 0.1 # smallest decimal place rod positions are rounded to in cm
    # MPC objective weights
    mpc_mv_weight: float = 5e-3  # movement penalty
    mpc_rate_weight: float = 10  # change rate penalty weight (c2)

    # Reproducibility
    seed: int = 108

    # Output control
    return_metrics: bool = False  # True during sensitivity runs; False for interactive use
    plot_results: bool = True  # suppressed during batch runs



#Class for the PID controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, kff, gain=1.0, integral_limit=10.0, proportional_limit=0.1, feedforward=False):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.gain = gain
        self.K_ff = kff

        # Anti-windup limits
        self.integral_limit = integral_limit
        self.proportional_limit = proportional_limit

        # State variables
        self.previous_error = 0.0
        self.previous_value = None
        self.previous_derivative = 0
        self.integral = 0.0
        self.last_time = None
        self.previous_setpoint = None


    def calc_feedforward(self, demand, current_react):
        #Calculates the steady state reactivity required to overcome the negative feedback from temperature at the demand level
        if self.previous_setpoint == None:
            self.previous_setpoint = demand
        goal_react = 5682 + 42 * demand #Linear relationship for steady state reactivity in feedforward term
        dr_dt = (demand - self.previous_setpoint) * 5000 #Demand change rate in feedforward term
        u_ff = (goal_react - current_react) * self.K_ff + dr_dt
        self.previous_setpoint = demand
        return u_ff

    def update(self, current_value, setpoint, current_react):
        # Calculate error terms
        error = setpoint - current_value
        delta_e = error - self.previous_error
        if self.previous_value == None:
            self.previous_value = current_value

        # Calculate time step
        dt = 2
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(
            min(self.integral, self.integral_limit), -self.integral_limit)

        # Derivative term
        new_derivative = (current_value - self.previous_value) / dt
        derivative = 0.2 * new_derivative + 0.8 * self.previous_derivative

        # PID output
        error = max(min(error, self.proportional_limit), -self.proportional_limit)

        control_signal = self.gain * \
                            (self.Kp * error + self.Ki * self.integral + self.Kd * derivative)



        self.previous_error = error
        self.previous_value = current_value
        self.previous_derivative = new_derivative
        return control_signal, delta_e


def rod_worth(x, rod):
    # Converts a rod position to a reactivity insertion in pcm using experimental rod worth curves
    if rod == 'AMDR':
        # Coefficients from experimental curve
        a0 = -0.074114
        a1 = 0.016447
        a2 = 0.012045
        a3 = -0.00006411
        critical_rod_worth = 0
    elif rod == 'SS1':
        a3 = -0.0401
        a2 = 3.8511
        a1 = -21.994
        a0 = 37.296
        critical_rod_worth = 0
    elif rod == 'SS2':
        a3 = -0.0203
        a2 = 2.0107
        a1 = -15.201
        a0 = 27.103
        critical_rod_worth = 0
    elif rod == 'RR':
        a3 = -0.0029
        a2 = 0.2693
        a1 = -0.5052
        a0 = 0.2674
        critical_rod_worth = 0
    react = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3  # Cubic function for rod worth
    return react - critical_rod_worth


def total_react(ss1_pos, ss2_pos, rr_pos, amdr_pos):
    # Calculates the total reactivity contribution from all 4 control rods
    return rod_worth(ss1_pos, 'SS1') + rod_worth(ss2_pos, 'SS2') \
           + rod_worth(rr_pos, 'RR') + rod_worth(amdr_pos, 'AMDR')

#Initializes the point kinetics model assuming previous steady state operation
def initialize_precursors(df):
    pur1 = PUR1Model(63, 47, 30, 30)
    for i in range(len(df)):
        df.loc[i, 'N_pke'] = df.loc[i, 'nfd-1-cps']
        precursors, T, I, Xe = pur1.calc_ss_concentration(df['Neutron Counts'].iloc[i])
        for j in range(6):
            name = 'C' + str(j + 1)
            df.loc[i, name] = precursors[j]
    return df


class PUR1Model:
    """
    Class for the code model of PUR1, includes many helper functions and stores control rod positions
    """

    def __init__(self, ss1_pos, ss2_pos, rr_pos, amdr_pos, discretization=0.01):
        # Rod Positions
        self.ss1_pos = ss1_pos
        self.ss2_pos = ss2_pos
        self.rr_pos = rr_pos
        self.amdr_pos = amdr_pos

        # Initial Conditions
        self.T = 293.15
        self.Xe = 0
        self.I = 0
        # Coefficients for experimental rod worth curve of AMDR
        self.amdr_a0 = -0.074114
        self.amdr_a1 = 0.016447
        self.amdr_a2 = 0.012045
        self.amdr_a3 = -0.00006411

        # Coefficients for experimental rod worth curve of RR
        self.rr_a0 = 0.2674
        self.rr_a1 = -0.5052
        self.rr_a2 = 0.2693
        self.rr_a3 = -0.0029

        # Rod Movement Speeds (cm/s)
        self.ss1_speed = SS_SPEED
        self.ss2_speed = SS_SPEED
        self.rr_speed = RR_SPEED
        self.amdr_speed = AMDR_SPEED
        self.discretization = discretization

        # Goal Reactivity: This is maintained inside the class in case an action will take more than one timestep
        self.goal_react = 5740
        self.C = np.array([0, 0, 0, 0, 0, 0])
        self.baseline_internal = 5682
        self.lookup_table_amdr = self.initialize_lookup('AMDR')
        self.lookup_table_rr = self.initialize_lookup('RR')
        self.lookup_table_ss2 = self.initialize_lookup('SS2')
        self.BETAS = BETAS * np.random.uniform(0.99, 1.01, 6)
        self.BETA = np.sum(self.BETAS)
        self.LAMBDA = LAMBDA * np.random.uniform(0.99, 1.01)
        self.LAMBDAS = LAMBDAS * np.random.uniform(0.99, 1.01, 6)
        return

    def pke(self, n0, rho_pcm_ex, C0):
        """
        Integrate the 6‐group PKE over t ∈ [0, 1] second,
        returning (n(1s), C(1s)).
        """
        T0 = self.T
        Xe0 = self.Xe / (counts_to_power_ratio * 100)
        I0 = self.I / (counts_to_power_ratio * 100)
        n0 = n0 / (counts_to_power_ratio * 100)
        C0 = C0 / (counts_to_power_ratio * 100)
        # Constants
        T_inf = 293.15
        alpha_T = -1.2
        Tau_T = 20

        #Iodine
        i_production = 0.063
        i_decay = 2.92E-5

        #Xenon
        xe_production = 0.0003
        xe_decay = 2.1066e-5
        sigma_xe = 2.6E-18
        rho_pcm_ex -= self.baseline_internal
        alpha_xe = -(5.2E-13 * counts_to_power_ratio)
        #Internal function for dynamics
        def rhs(t, y):
            n = y[0]
            C = y[1:7]
            T = y[7]
            I = y[8]
            Xe = y[9]
            rho_t = alpha_T * (T - T_inf)
            rho_x = alpha_xe * Xe
            rho_pcm = rho_pcm_ex + rho_t + rho_x
            unitless_rho = rho_pcm * 1e-5
            # dn/dt (Change in neutron counts)

            dn_dt = ((unitless_rho - self.BETA) / self.LAMBDA) * n + self.LAMBDAS.dot(C) + SOURCE
            # dCi/dt (Change in precursor concentration)
            dC_dt = (self.BETAS / self.LAMBDA) * n - self.LAMBDAS * C
            # dT/dt (Change in temperature)
            dT_dt = (((35 * n) + T_inf) - T) / Tau_T
            # dI/dt (Change in Iodine)
            dI_dt = i_production * n - i_decay * I
            # dXe/dt (Change in Xenon)
            dXe_dt = xe_production * n + i_decay * I - (xe_decay + sigma_xe * n) * Xe
            return np.concatenate([[dn_dt], dC_dt, [dT_dt], [dI_dt], [dXe_dt]])

        y0 = np.concatenate([[n0], C0, [T0], [I0], [Xe0]])
        sol = solve_ivp(rhs, [0, 1.0], y0, method='LSODA', t_eval=[1.0])
        y1 = sol.y[:, -1]
        y1[0:7] = y1[0:7] * counts_to_power_ratio * 100
        y1[8] = y1[8] * counts_to_power_ratio * 100
        y1[9] = y1[9] * counts_to_power_ratio * 100
        self.T = y1[7]
        self.I = y1[8]
        self.Xe = y1[9]
        return y1[0], y1[1:7]

    def rod_worth(self, x, rod):
        # Converts a rod position to a reactivity insertion in pcm using experimental rod worth curves
        if rod == 'AMDR':
            # Coefficients from experimental curve
            a3 = self.amdr_a3
            a2 = self.amdr_a2
            a1 = self.amdr_a1
            a0 = self.amdr_a0
            critical_rod_worth = 0
        elif rod == 'SS1':
            a3 = -0.0401
            a2 = 3.8511
            a1 = -21.994
            a0 = 37.296
            critical_rod_worth = 0
        elif rod == 'SS2':
            a3 = -0.0203
            a2 = 2.0107
            a1 = -15.201
            a0 = 27.103
            critical_rod_worth = 0
        elif rod == 'RR':
            a3 = -0.0029
            a2 = 0.2693
            a1 = -0.5052
            a0 = 0.2674
            critical_rod_worth = 0
        react = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3  # Cubic function for rod worth
        return react - critical_rod_worth

    def initialize_lookup(self, rod):
        #Builds a lookup table to avoid inverse cubic solution anytime we go from reactivity to rod position
        x_vals = np.linspace(0, 64, 10000)
        y_vals = rod_worth(x_vals, rod)
        inverse_cubic_interp = interp1d(y_vals, x_vals, kind='linear', bounds_error=False, fill_value='extrapolate')
        return inverse_cubic_interp

    def find_rod_react(self, goal_react, rod):
        # Finds the amount of reactivity that needs to be introduced from any rod to reach the goal reactivity
        if rod == 'AMDR':
            other_react = self.rod_worth(self.ss1_pos, 'SS1') + self.rod_worth(self.ss2_pos, 'SS2') + self.rod_worth(
                self.rr_pos, 'RR')
        elif rod == 'RR':
            other_react = self.rod_worth(self.ss1_pos, 'SS1') + self.rod_worth(self.ss2_pos, 'SS2') + self.rod_worth(
                self.amdr_pos, 'AMDR')
        elif rod == 'SS2':
            other_react = self.rod_worth(self.ss1_pos, 'SS1') + self.rod_worth(self.rr_pos, 'RR') + self.rod_worth(
                self.amdr_pos, 'AMDR')
        return goal_react - other_react

    def calc_ss_concentration(self, n0):
        # Calculates the steady state precursor concentraion in the PKE model given the starting neutron counts
        # C_i = (n0 * β_i) / (Λ * λ_i)
        i_production = 0.063
        i_decay = 2.92E-5

        xe_production = 0.0003
        xe_decay = 2.1066e-5
        fissions = (n0 / (counts_to_power_ratio * 100)) * 3.12E14 / 5700
        sigma_xe = (2.6e6) * 1e-24
        C = (n0 * self.BETAS) / (self.LAMBDA * self.LAMBDAS)
        I0 = i_production * fissions / i_decay
        Xe0 = (xe_production * fissions + i_decay * I0 - sigma_xe * n0) / xe_decay
        T0 = 293.15 + 35 * n0 / (counts_to_power_ratio * 100)
        return C, I0, Xe0, T0

    def total_react(self):
        # Calculates the total reactivity contribution from all 4 control rods
        return self.rod_worth(self.ss1_pos, 'SS1') + self.rod_worth(self.ss2_pos, 'SS2') \
               + self.rod_worth(self.rr_pos, 'RR') + self.rod_worth(self.amdr_pos, 'AMDR')

    def move_rod(self, rod, new_position):
        """
        Function that moves the control rods
        Limitations are:
        1. Rod cannot move faster than its movement speed
        2. Rod cannot move outside of its physical range
        3. Rod position cannot be known at finer resolution than 0.1cm (0.5cm for AMDR)
        """

        if rod == 'ss1':
            movement_amount = new_position - self.ss1_pos
            movement_amount = np.clip(movement_amount, -self.ss1_speed, self.ss1_speed)
            self.ss1_pos += movement_amount
            self.ss1_pos = np.float32(np.clip(self.ss1_pos, 0, 64))
            self.ss1_pos = np.round(self.ss1_pos, decimals=2)
        elif rod == 'ss2':
            movement_amount = new_position - self.ss2_pos
            movement_amount = np.clip(movement_amount, -self.ss2_speed, self.ss2_speed)
            self.ss2_pos += movement_amount
            self.ss2_pos = np.float32(np.clip(self.ss2_pos, 0, 64))
            self.ss2_pos = np.round(self.ss2_pos, decimals=2)
        elif rod == 'rr':
            movement_amount = new_position - self.rr_pos
            movement_amount = np.clip(movement_amount, -self.rr_speed, self.rr_speed)
            self.rr_pos += movement_amount
            self.rr_pos = np.float32(np.clip(self.rr_pos, 0, 64))
            self.rr_pos = round(self.rr_pos / self.discretization) * self.discretization
        elif rod == 'amdr':
            # AMDR Moves faster, rounds to the nearest 0.5cm instead of 0.1cm
            movement_amount = new_position - self.amdr_pos
            movement_amount = np.clip(movement_amount, -self.amdr_speed, self.amdr_speed)
            self.amdr_pos += movement_amount
            self.amdr_pos = np.float32(np.clip(self.amdr_pos, 0, 64))
            self.amdr_pos = np.round(self.amdr_pos, decimals=1)
        return

    def update_C(self, df, index):
        #Updates the precursor concentration
        for j in range(len(self.C)):
            name = 'C' + str(j + 1)
            df.loc[index + 1, name] = self.C[j]
        return df

    def simulate(self, df, action_horizon, initialize):
        #This is the function that runs each iteration and propagates the true plant state forward in time
        if initialize:
            self.C, self.I, self.Xe, self.T = self.calc_ss_concentration(df.loc[INPUT_TIME - 1, 'Neutron Counts'])
            df = self.update_C(df, 0)

        else:
            for j in range(len(self.C)):
                name = 'C' + str(j + 1)
                # self.C[j] = df.loc[INPUT_TIME, name]
            df = self.update_C(df, INPUT_TIME - 1)
        for i in range(action_horizon + 1):
            index = INPUT_TIME + i - 1
            self.ss1_pos = df.loc[index, 'ss1-position']
            self.ss2_pos = df.loc[index, 'ss2-position']
            self.rr_pos = df.loc[index, 'rr-position']
            self.amdr_pos = df.loc[index, 'amdr-position']
            react = self.total_react()
            n_new, self.C = self.pke(df.loc[index, 'Neutron Counts'], react, self.C)
            df.loc[index + 1, 'Neutron Counts'] = n_new
            df.loc[index + 1, 'nfd-1-cps'] = n_new / (counts_to_power_ratio * CHANNEL_1_TO_2)
            df.loc[index + 1, 'nfd-2-log'] = n_new / (counts_to_power_ratio)
            df = self.update_C(df, index)
        return df


class ControlModule():
    '''
    Class for hosting the PID and MPC, along with MHE and helper functions
    '''
    def __init__(self, input_data, beta, l1, lamb, mhe_window, dcost, ccost, discretization, kp, ki, kd, kff, mismatch, ff):
        self.C = None
        #Perturbed parameters from dataclass in Sensitivity Analysis
        self.beta = beta * mismatch
        self.l1 = l1*mismatch
        self.lamb = lamb*mismatch
        #Values for MPC
        self.mhe_window = mhe_window
        self.dcost = dcost
        self.ccost = ccost
        self.discretization = discretization
        ###
        self.data = input_data
        self.PUR1 = PUR1Model(63, 47, 30, 30, discretization=self.discretization)
        self.columns = ['index', 'nfd-1-cr', 'nfd-1-cps', 'rr-active-state', 'rr-position', 'ss1-active-state',
                        'ss1-position',
                        'ss2-active-state', 'ss2-position', 'amdr-active-state', 'amdr-position', 'group_id']
        #Gain, integral clamp, and proportional limit for saturation are set here
        self.current_pid = PIDController(Kp=kp, Ki=ki, Kd=kd, kff=kff, gain=40, integral_limit=10, proportional_limit=0.02, feedforward=ff)
        self.feedforward = ff
        #Starting guess for excess reactivity, MHE will refine this guess
        self.baseline_reactivity = starting_baseline
        self.demand = None

        return

    def update(self, input_data):
        '''
        Updates the ControllerModule class with new input data and updates the rods on the PUR1 class
        '''

        self.data = input_data
        data = self.data
        self.PUR1.ss1_pos = data['ss1-position'].iloc[-1]
        self.PUR1.ss2_pos = data['ss2-position'].iloc[-1]
        self.PUR1.rr_pos = data['rr-position'].iloc[-1]
        self.PUR1.amdr_pos = data['amdr-position'].iloc[-1]
        return

    def react_based_position(self, potential_react, rod):
        #Same as function included in PUR1 class
        if rod == 'AMDR':
            rod_react = np.clip(self.PUR1.find_rod_react(potential_react, rod), 1, 25)
            rod_position = self.PUR1.lookup_table_amdr(rod_react)
        elif rod == 'RR':
            rod_react = np.clip(self.PUR1.find_rod_react(potential_react, rod), 10, 300)
            rod_position = self.PUR1.lookup_table_rr(rod_react)
        elif rod == 'SS2':
            rod_react = np.clip(self.PUR1.find_rod_react(potential_react, rod), 50, 2000)
            rod_position = self.PUR1.lookup_table_ss2(rod_react)
        return rod_position

    def make_cr_movements(self, rod, goal_position):
        # Same as function included in PUR1 class
        goal_position = round(goal_position / self.discretization) * self.discretization
        if rod == 'AMDR':
            current_position = self.PUR1.amdr_pos
        elif rod == 'RR':
            current_position = self.PUR1.rr_pos
        elif rod == 'SS2':
            current_position = self.PUR1.ss2_pos
        rod_positions = []
        active_states = []
        if rod == 'AMDR':
            rod_speed = AMDR_SPEED
        elif rod == 'RR':
            rod_speed = RR_SPEED
        elif rod == 'SS2' or rod == 'SS1':
            rod_speed = SS_SPEED
        for i in range(OUTPUT_TIME):
            new_position = np.clip(goal_position, current_position - rod_speed, current_position + rod_speed)
            if new_position > (current_position + 0.1):
                active_states.append(1)
            elif new_position < (current_position - 0.1):
                active_states.append(-1)
            else:
                active_states.append(0)
            rod_positions.append(current_position)
            current_position = new_position
        return rod_positions, active_states

    def form_df(self, rod_positions, active_states, rod):
        #Functino for rebuilding the dataframe including the rod positions and active state generated from the output control action
        columns = self.data.columns
        zeros = np.zeros((OUTPUT_TIME, len(columns)))
        new_df = pd.DataFrame(zeros, columns=columns)

        # Setting future rod positions to remain at current values
        new_df['ss1-position'] = self.data['ss1-position'].iloc[-1]
        new_df['ss2-position'] = self.data['ss2-position'].iloc[-1]
        if rod == 'AMDR':
            new_df['amdr-position'] = rod_positions
            new_df['amdr-active-state'] = active_states
            new_df['rr-position'] = self.data['rr-position'].iloc[-2]
            new_df['rr-active-state'] = 0
        elif rod == 'RR':
            new_df['rr-position'] = rod_positions
            new_df['rr-active-state'] = active_states
            new_df['amdr-position'] = self.data['amdr-position'].iloc[-1]
            new_df['amdr-active-state'] = 0
        elif rod == 'SS2':
            new_df['ss2-position'] = rod_positions
            new_df['ss2-active-state'] = active_states
            new_df['amdr-position'] = self.data['amdr-position'].iloc[-1]
            new_df['amdr-active-state'] = 0

        # Setting future rod active states to remain at zero
        active_states = ['ss1-active-state']
        for rod in active_states:
            new_df[rod] = 0

        complete_df = pd.concat([self.data, new_df])
        complete_df.reset_index(inplace=True, drop=True)
        complete_df['index'] = complete_df.index
        complete_df['group_id'] = 1
        for i in range(len(complete_df)):
            react = total_react(complete_df.loc[i, 'ss1-position'], complete_df.loc[i, 'ss2-position'],
                                complete_df.loc[i, 'rr-position'], complete_df.loc[i, 'amdr-position'])
            complete_df.loc[i, 'Reactivity (pcm)'] = react
            if i != 0:
                complete_df.loc[i, 'Reactivity change rate'] = react - complete_df.loc[i - 1, 'Reactivity (pcm)']
        complete_df.loc[0, 'Reactivity change rate'] = 0
        # complete_df.loc[:, 'Reactivity (pcm)'] += MODEL_REACT_OFFSET
        return complete_df

    def calc_change_rate(self, data_complete):
        #Function for handmade recreation of reactor change rate. We couldn't find exactly how nfd-1-cr is calculated
        #in the reactor handbooks, we know it has some filtering and clamping. This was our best approximation
        for i in range(len(data_complete) - 6):
            avg_1 = data_complete.loc[i:i + 5, 'nfd-1-cps'].mean()
            avg_2 = data_complete.loc[i + 1:i + 6, 'nfd-1-cps'].mean()
            cr = max(min(100 * (avg_2 - avg_1) / (avg_1 + 40), 8), -3)
            data_complete.loc[i + 6, 'nfd-1-cr'] = cr

        return data_complete

    def remake_df(self, df, predictions, feature):
        #Function for adding the neutron predictions to the rebuilt df
        df.loc[(INPUT_TIME):(INPUT_TIME + OUTPUT_TIME), feature] = predictions

        df['nfd-1-cps'] = df['Neutron Counts'] / (counts_to_power_ratio * CHANNEL_1_TO_2)
        df['N_pke'] = df['Neutron Counts']
        df['nfd-2-log'] = df['Neutron Counts'] / counts_to_power_ratio
        return df

    def MPC_v2(self, current_n, current_setpoint, current_react,
               control_horizon, prediction_horizon, dcost, ccost, integral_penalty):
        '''
        Function that builds and solves the MPC problem, uses a coldstart with 3 nodes

        '''
        n0 = current_n
        c10 = self.C
        sp = current_setpoint

        # Build GEKKO MPC Model
        mpc_dir = 'mpc_clean'
        os.makedirs(mpc_dir, exist_ok=True)
        t0 = time.perf_counter()
        m = GEKKO(name='pke_mpc', remote=False)
        m.path = mpc_dir
        #Setting solver options
        m.options.NODES = 3
        m.options.IMODE = 6  # MPC mode
        m.options.SOLVER = 3
        m.options.DIAGLEVEL = 1
        m.options.CTRL_HOR = 2

        # time horizon
        H = prediction_horizon
        m.time = np.linspace(0, H, H + 1)
        #physical constants
        b1 = m.Const(value=self.beta)
        lamb = m.Const(value=self.lamb)
        l1 = m.Const(value=self.l1)
        #Reactivity as the manipulated variable
        rho = m.MV(value=current_react, lb=-20, ub=20)
        rho.STATUS = 1
        rho.FSTATUS = 1
        rho.DCOST = dcost  # cost for movement of MV
        rho.DMAX = 10  # max movement per control interval
        #Neutron counts as the controlled variable
        n = m.CV(value=n0, lb=0)
        n.STATUS = 1
        n.FSTATUS = 1
        n.MEAS = n0  # incoming measurement
        n.SP = sp
        m.options.CV_TYPE = 2  # quadratic error

        c1 = m.Var(value=c10, lb=0)

        # terminal penalty:
        p = np.zeros(len(m.time))
        p[-1] = 1.0  # only active at final time point
        final = m.Param(value=p)
        m.Minimize(final * (n - sp) ** 2)

        #Neutron equation
        m.Equation(n.dt() == n * ((rho * 1e-5 - b1) / lamb) + (c1 * l1))

        # Precursor equation
        m.Equation(c1.dt() == (n * b1 / lamb) - l1 * c1)

        dn = m.Var(value=0)
        m.Equation(dn == n.dt())  # define derivative as a variable and add to minimization
        m.Minimize(ccost * dn ** 2)

        try:
            m.solve(disp=False)
        except:
            print("MPC infeasible this cycle!")
            return current_react, current_n

        predicted_n = np.array(n.value) * counts_to_power_ratio * 100
        rho_out = float(rho.value[1])
        return rho_out, predicted_n[1:11]

    def run_control_mpc(self, input_data, demand, rod, packet_lost):
        # Control Function for Model Predictive Control
        self.update(input_data)  # Update the class internal values based on the received input data
        # Loop to add in the external reactivity values to the input data based on rod positions
        if self.demand is not None:
            previous_demand = self.demand
        self.demand = demand
        #Filling in the dataframe with the correctly calculated reactivity values
        for i in range(len(input_data)):
            react = total_react(input_data.loc[i, 'ss1-position'], input_data.loc[i, 'ss2-position'],
                                input_data.loc[i, 'rr-position'], input_data.loc[i, 'amdr-position'])
            input_data.loc[i, 'Reactivity (pcm)'] = react

        n_values = input_data['Neutron Counts'].values / (100 * counts_to_power_ratio)
        current_n = n_values[-1]

        self.precursor_estimator(current_n)
        t = time.perf_counter()
        if not packet_lost:
            try:
                rho_estimated = self.MHE_estimator_v2(n_values)
                prev_baseline = self.baseline_reactivity
                #Adjusting the excess reactivity slowly based on MHE outputs to avoid noise or spikes on control movement
                self.baseline_reactivity = 0.05 * (
                        input_data['Reactivity (pcm)'].iloc[-1] - (rho_estimated)) + prev_baseline * 0.95

            except:
                print('MHE Failed to find solution')
                rho_estimated = 0
        else:
            #Linear feedback fallback, same as PID feedforward term if there is no measured data
            self.baseline_reactivity = 0.9 * self.baseline_reactivity + 0.1 * (5682.5 + 42 * current_n)

        mhe_elapsed = time.perf_counter() - t
        current_rho = input_data['Reactivity (pcm)'].iloc[-1] - self.baseline_reactivity
        control_horizon = 2
        prediction_horizon = 15
        ccost = self.ccost
        dcost = self.dcost * np.mean(n_values)
        icost = 0
        Cs = []
        for i in range(6):
            name = 'C' + str(i + 1)
            Cs.append(input_data[name].iloc[-1])
        #Running MPC
        t = time.perf_counter()
        rho, n = self.MPC_v2(current_n, demand, current_rho, control_horizon, prediction_horizon, dcost, ccost, icost)
        mpc_elapsed = time.perf_counter() - t
        # rho, n = rho[-1], n[:OUTPUT_TIME]
        # if abs(rho) <= 1:
        #    rho = 0

        position = self.react_based_position(rho + self.baseline_reactivity, rod)

        #This is all for rebuilding the dataframe as the code logic is based off of having an output dataframe with
        # OUTPUT_SIZE seconds of future predicted data appended to the end. This is not really necessary, but is based
        # on previous implementations of this code for other purposes
        rod_positions, active_states = self.make_cr_movements(rod, position)
        rod_positions = np.array(rod_positions)
        rod_positions = np.round(rod_positions / self.discretization) * self.discretization
        rod_positions = rod_positions * np.random.uniform(0.9999, 1.0001)
        complete_df = self.form_df(rod_positions, active_states, rod)

        complete_df = self.remake_df(complete_df, n, 'Neutron Counts')
        # complete_df = self.unscale_data(complete_df, self.scaler)
        complete_df = self.calc_change_rate(complete_df)
        best_prediction = complete_df
        best_action = position
        return best_prediction, best_action, mhe_elapsed, mpc_elapsed


    def run_control_pid(self, df, demand, rod, method, packet_lost):
        '''
        Main function for running the PID controller either with or without feedforward
        '''
        self.update(df)
        input_data = self.data
        for i in range(len(input_data)):
            react = total_react(input_data.loc[i, 'ss1-position'], input_data.loc[i, 'ss2-position'],
                                input_data.loc[i, 'rr-position'], input_data.loc[i, 'amdr-position'])
            input_data.loc[i, 'Reactivity (pcm)'] = react
        current_n = input_data['Neutron Counts'].iloc[-1] / (100 * counts_to_power_ratio)
        current_reactivity = input_data['Reactivity (pcm)'].iloc[-1]
        if method == 'current' and not packet_lost:
            control_signal, delta_e = self.current_pid.update(current_n, demand, current_react=current_reactivity)
            control_signal = control_signal / max(current_n, demand)
        else:
            control_signal = 0
        if self.feedforward:
            control_signal += self.current_pid.calc_feedforward(demand, current_reactivity)
        if -0.5 < control_signal < 0.5:
            control_signal = 0

        if method == 'baseline':
            position = self.react_based_position(control_signal + self.baseline_reactivity, rod)
        elif method == 'current':
            position = self.react_based_position(control_signal + current_reactivity, rod)
        if control_signal != 0:
            position = position * np.random.uniform(0.9999, 1.0001)
        rod_positions, active_states = self.make_cr_movements(rod, position)
        complete_df = self.form_df(rod_positions, active_states, rod)
        best_action = position
        return complete_df, best_action

    def MHE_estimator_v2(self, n_values):
        '''
        Function for the Moving horizon estimator used to calculate the excess reactivity from the data
        '''
        n_values = n_values[INPUT_TIME - 1 - self.mhe_window:INPUT_TIME - 1] / (counts_to_power_ratio)
        # Time horizon and steps
        timesteps = self.mhe_window  # Number of MHE time steps (10s by default)
        dt = 1  # Time step (1 second)
        m = GEKKO(remote=False)
        m.time = np.linspace(0, (timesteps - 1) * dt, timesteps)

        # MHE setup
        m.options.IMODE = 5
        m.options.CV_TYPE = 1
        m.options.DIAGLEVEL = 0
        m.options.NODES = 3
        #Constansts
        b1 = m.Const(value=self.beta)
        lamb = m.Const(value=self.lamb)
        l1 = m.Const(value=self.l1)

        # Measured input (reactor power)
        n = m.CV(value=n_values)
        c = m.Var(value=self.C)
        n.STATUS = 0  # measured
        n.FSTATUS = 1  # measured input
        n.WMEAS = 1e-2  # weight for measurement matching
        n.MEAS_GAP = 1
        n.WSP = 100

        # Reactivity
        rho = m.MV()
        rho.STATUS = 1
        rho.FSTATUS = 0
        rho.DCOST = 30
        rho.LOWER = -2e-4
        rho.UPPER = 2e-4

        dn = m.Var()
        # Neutron equation:
        m.Equation(n.dt() == n * ((rho * 1e-5 - b1) / lamb) + (c * l1))

        # Precursor equation:
        m.Equation(c.dt() == (n * b1 / lamb) - l1 * c)
        #Change rate penalty
        m.Equation(dn == n.dt())
        m.Obj(10 * dn ** 2)
        # Solve MHE
        m.solve(disp=False)
        return np.mean(rho.value) * 1e5

    def precursor_estimator(self, n):
        #Internal one-group precursor estimation
        if self.C is None:
            self.C = n * self.beta / (self.lamb * self.l1)
        else:
            dC = n * self.beta / self.lamb - self.l1 * self.C
            self.C += dC * STEP_SIZE
        return


def demo_load_data(i):
    #function for loading the initial data to begin the control experiment, this sets up the dataframe and columns
    column_names = ["nfd-1-cps", "nfd-1-cr", "nfd-2-log", "nfd-2-cr", "nfd-3-pwr", "nfd-4-flux", "rr-position",
                    "ss1-position", "ss2-position",
                    "rr-active-state", "ss1-active-state", "ss2-active-state", "ram-pool-lvl", "ram-con-lvl",
                    "ram-wtr-lvl",
                    "cont air counts"]

    zeros = np.zeros((40, 16))
    df = pd.DataFrame(zeros, columns=column_names)
    df = df.dropna()

    df = df.head(INPUT_TIME)
    df = df.reset_index()

    df = df[:INPUT_TIME]
    df['Reactivity (pcm)'] = 0.0
    df['Reactivity change rate'] = 0.0
    df['C1'] = 0.0
    df['C2'] = 0.0
    df['C3'] = 0.0
    df['C4'] = 0.0
    df['C5'] = 0.0
    df['C6'] = 0.0
    df['PKE_Prediction'] = 0.0
    df['amdr-position'] = AMDR_START_POS
    df['amdr-active-state'] = 0.0
    return df


def demand_generation(method, total_time):
    #Function for generating the demand time series based on the string description of the scenario
    if method == '3%-2%-3% (AMDR)':
        demand = np.ones(total_time)
        demand[0:600] = 0.03
        demand[600:1800] = 0.02
        demand[1800:total_time] = 0.03
    elif method == 'step70-80':
        demand = np.ones(total_time)
        demand[0:200] = 0.7
        demand[200:800] = 0.8
        demand[800:total_time] = 0.7
    elif method == '3% to 80% (RR)':
        demand = np.ones(total_time)
        demand[0:600] = 0.03
        demand[600:total_time] = 0.8
    elif method == '10%-30% Ramp (RR)':
        demand = np.ones(total_time)
        demand[0:200] = 0.1
        demand[200:1100] = np.linspace(start=0.1, stop=0.3, num=900)
        demand[1100:1700] = 0.3
        demand[1700:2600] = np.linspace(start=0.3, stop=0.1, num=900)
        demand[2600:total_time] = 0.1
    elif method == '70% Steady State':
        demand = np.ones(total_time)
        demand[0:total_time] = 0.7
    elif method == '70% Steady State with ramp':
        demand = np.ones(total_time)
        demand[0:1000] = 0.7
        demand[1000:2000] = np.linspace(start=0.7, stop=0.8, num=1000)
        demand[2000:total_time] = 0.8
    elif method == 'Training Data':
        demand = np.ones(total_time)
        demand[0:2000] = 0.7
        indices = np.random.rand(200) * 0.5 + 0.3
        for i in range(200):
            start = i * 500
            end = (i + 1) * 500
            demand[start:end] = indices[i]
    # demand = demand + ((2*np.random.rand()-1)/100)
    return demand


def make_starting_data(df, starting_power, ss1_position, ss2_position, rr_position, amdr_position):
    #Second function for initial dataframe, this one corrects the rod positions and starting power
    df['nfd-2-log'] = starting_power
    df['nfd-1-cps'] = df['nfd-2-log'] / CHANNEL_1_TO_2
    df['Neutron Counts'] = df['nfd-1-cps'] * counts_to_power_ratio * CHANNEL_1_TO_2
    df['ss1-position'] = ss1_position
    df['ss2-position'] = ss2_position
    df['rr-position'] = rr_position
    df['amdr-position'] = amdr_position
    for i in range(len(df)):
        react = total_react(df.loc[i, 'ss1-position'], df.loc[i, 'ss2-position'],
                            df.loc[i, 'rr-position'], df.loc[i, 'amdr-position'])
        df.loc[i, 'Reactivity (pcm)'] = react
    return df


def add_noise(df, features, noise_stds):
    #Function for adding gaussian noise to the signals
    for i in range(len(noise_stds)):
        df[features[i]] = df[features[i]] + np.random.normal(0, noise_stds[i], size=len(df))
    return df


def calc_metrics(df):
    #Function for calculating control effort and RMSE after the script finished
    #NOTE: noise is filtered out of the control effort calculation, so any rod movement less than 0.01 cm is not added.
    #The controllers are only allowed to move the the rod in 0.1cm increments so this will not filter out real actions
    threshold = 0.01
    rmse = np.sqrt(np.mean((df['Power'] - df['Demand']) ** 2))
    control_effort_diffs = df['rr-position'].diff().abs()
    control_effort = control_effort_diffs[control_effort_diffs > threshold].sum()
    print('RMSE:',rmse)
    print('Control Effort:', control_effort)
    return rmse, control_effort


def generate_packet_losses(total_packets, p_loss=0.01, burst_len=1500, burst_prob=0.003, seed=None):
    '''
    Function for generating the sequence of timesteps where there is packet loss. This includes both the burst packet
    loss which uses a Markov chain and the stochastic packet loss which is an independent probability
    '''
    #Setting up the independent packet loss
    rng = np.random.default_rng(seed)
    independent_losses = rng.random(total_packets) < p_loss
    independent_indices = np.where(independent_losses)[0]
    health = True
    burst_indices = []
    starts = []
    lens = []
    #Burst packet loss
    for i in range(total_packets):
        roll = rng.random()
        prev_state = health
        if health:
            if roll < (burst_prob):
                health = False
            else:
                health = True
        else:
            if roll > (1 / (burst_len / STEP_SIZE)):
                health = False
            else:
                health = True

        if not health:
            burst_indices.append(i)
            if prev_state:
                starts.append(i)
            if i == total_packets - 1:
                len = i - starts[-1]
                lens.append(len)
        elif not prev_state:
            len = i - starts[-1]
            lens.append(len)

    #Combine and dedupe
    all_losses = np.sort(np.unique(np.concatenate([independent_indices, burst_indices])))
    print(starts, lens)
    return all_losses, starts, lens


def main_pke(cfg: SimConfig = None) -> Optional[dict]:
    '''
    This is the main function which runs the control experiment and is called in the sensitivity analysis code
    '''
    #Loading the dataclass if it exists
    if cfg is None:
        cfg = SimConfig()  # default interactive behaviour unchanged

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    i = 0

    # Initializing the measured data, true data, and previous measured data
    df = demo_load_data(i)
    measured_data = make_starting_data(df, starting_power=70, ss1_position=62.58, ss2_position=46.88, rr_position=30.8,
                                       amdr_position=0)
    old_measured_data = measured_data
    true_data = measured_data

    # Initializing Controller, diagnostics, and PUR1 classes based on dataclass
    controller = ControlModule(df, beta=BETA * cfg.beta_scale, l1=L1 * cfg.lambda_scale,
                               lamb=LAMBDA * cfg.lambda_scale, mhe_window=cfg.mhe_window, dcost=cfg.mpc_mv_weight,
                               ccost=cfg.mpc_rate_weight, discretization=cfg.discretization, kp=cfg.kp, ki=cfg.ki, kd=cfg.kd, kff=cfg.kff, mismatch=cfg.beta_scale, ff=cfg.feedforward)
    pur1 = PUR1Model(62.58, 46.88, 30.8, 0, discretization=cfg.discretization)
    # Initializing saved results .csv
    saved_results = pd.DataFrame(
        columns=['nfd-1-cps', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state',
                 'ss2-position', 'amdr-position', 'nfd-1-cr', 'Time',
                 'Demand', 'Power', 'Neutron Counts'])

    demand_curve = demand_generation(cfg.demand_scenario, cfg.num_iter * (STEP_SIZE))
    # Initializing operation states
    rod = 'RR'
    open_loop = False #Open loop variable is not changed in this code, relic from previous iteration
    secondary_sensor = False

    # Terms for packet loss
    pl_list, starts, lens = generate_packet_losses(cfg.num_iter, p_loss=cfg.stochastic_threshold,
                                                   burst_len=cfg.burst_len,
                                                   burst_prob=cfg.burst_prob, seed=cfg.seed)

    #making lists used for tracking some variables
    tracked_measured_power = []
    total_runtimes = []
    mpc_runtimes = []
    mhe_runtimes = []
    mpc_predictions = []

    for i in range(cfg.num_iter - 1):
        # Setting demand based on demand curve
        demand = demand_curve[i*STEP_SIZE]
        # Setting Packet loss based on random sequence
        if i in pl_list and i > 30:
            packet_lost = True
        else:
            packet_lost = False
        # Setting measured data to be best prediction if no measured data available
        if i > 0:
            if packet_lost or open_loop:
                measured_data = best_prediction.iloc[STEP_SIZE:INPUT_TIME + STEP_SIZE, :].copy()
            else:
                # If measured data is available, setting measured data to be true data plus noise and latency
                measured_data = true_data.iloc[STEP_SIZE:INPUT_TIME + STEP_SIZE, :].copy()
                latency = 1

                # Model 1-second latency in nfd-1-cps: measured value is always 1 step behind true data
                measured_data['nfd-1-cps'] = true_data['nfd-1-cps'].iloc[
                                             STEP_SIZE - latency:INPUT_TIME + STEP_SIZE - latency
                                             ].values
                measured_data['nfd-2-log'] = true_data['nfd-2-log'].iloc[
                                             STEP_SIZE - latency:INPUT_TIME + STEP_SIZE - latency
                                             ].values


            true_data = true_data.iloc[STEP_SIZE:INPUT_TIME + STEP_SIZE, :].copy()

        # Resetting the index
        measured_data = measured_data.reset_index(drop=True)

        # Adding Noise
        noise_features = ['nfd-1-cps', 'nfd-2-log', 'ss1-position', 'ss2-position', 'rr-position', 'amdr-position']
        noise_stds = np.array(
            [0.002 * measured_data['nfd-1-cps'].iloc[-1], 0.002 * measured_data['nfd-2-log'].iloc[-1], 0.001, 0.001,
            0.001, 0.001])
        measured_data = add_noise(measured_data, noise_features, noise_stds)
        # Swtich to secondary sensor if needed, not used here
        if secondary_sensor:
            measured_data['Neutron Counts'] = measured_data['nfd-2-log'] * counts_to_power_ratio
        else:
            measured_data['Neutron Counts'] = measured_data['nfd-1-cps'] * CHANNEL_1_TO_2 * counts_to_power_ratio

        # Assigning index and group_id (group_id not used here)
        measured_data['index'] = measured_data.index
        measured_data['group_id'] = 1

        # Starting timer to extract runtime
        ##### Beginning of Autonomous Loop #####
        if i == 0:
            measured_data = initialize_precursors(df)
        t = time.perf_counter()
        #Running the controller based on the method chosen in the dataclass
        if cfg.method == 'MPC':
            best_prediction, best_action, mhe_time, mpc_time = controller.run_control_mpc(measured_data, demand, rod, packet_lost)
        elif cfg.method == 'PID':
            best_prediction, best_action = controller.run_control_pid(measured_data, demand, rod, 'current',
                                                                      packet_lost)
        elapsed = time.perf_counter() - t
        total_runtimes.append(elapsed)
        if cfg.method == 'MPC':
            #Append individual runtimes and predictions only if using MPC
            mhe_runtimes.append(mhe_time)
            mpc_runtimes.append(mpc_time)
            mpc_pred = best_prediction['Neutron Counts'].iloc[-1]
            mpc_predictions.append(mpc_pred)
        ##### End of Autonomous Loop #####

        ##### Applying Actions to PKE model #####
        control_signals = ['ss1-position', 'ss1-active-state', 'ss2-position', 'ss2-active-state', 'rr-position',
                           'rr-active-state', 'amdr-position', 'amdr-active-state']
        true_data[control_signals] = best_prediction[control_signals]
        true_data.reset_index(drop=True)
        best_prediction.reset_index(drop=True)
        # Concatenated the predicted data and outputs on the true data
        # NOTE: THE MEASURED PARAMETERS (NEUTRON COUNTS) ARE OVERWRITTEN IN THE TRUE DATA, TRUE DATA IS NOT AFFECTED BY MPC PREDICTIONS
        true_data = pd.concat([true_data, best_prediction[INPUT_TIME:INPUT_TIME + STEP_SIZE]], ignore_index=True)
        true_data.reset_index(drop=True)
        # Initializing PKE if just starting simulation
        if i == 0:
            true_data = pur1.simulate(true_data, STEP_SIZE - 1, True)
        else:
            true_data = pur1.simulate(true_data, STEP_SIZE - 1, False)
        # Reinitializing internal precursor concentration (assuming infinite lookback at steady state) if measurements are restored and current power
        # predictions are off by more than 5%
        if not packet_lost and not open_loop:
            if abs(measured_data['Neutron Counts'].iloc[-1] - true_data['Neutron Counts'].iloc[-1]) / (
                    100 * counts_to_power_ratio) > 0.05:
                controller.precursor_estimator(true_data['Neutron Counts'].iloc[-1] / (100 * counts_to_power_ratio))
            measured_data = true_data
        else:
            measured_data = best_prediction.loc[STEP_SIZE:INPUT_TIME + STEP_SIZE, :]


        # Adding derived signal of neutron change rate to both sets of data
        measured_data = controller.calc_change_rate(measured_data)
        true_data = controller.calc_change_rate(true_data)

        measured_data['N_pke'] = measured_data['Neutron Counts']


        ##### Move to next timestep #####
        tracked_measured_power.append(
            measured_data.loc[INPUT_TIME - 1, 'nfd-1-cps'] * CHANNEL_1_TO_2 / 100
            if not open_loop
            else tracked_measured_power[-1] * (1 + i * 0.01))  # continues drifting after open loop switch
        tracked_measured_power.append(
            measured_data.loc[INPUT_TIME, 'nfd-1-cps'] * CHANNEL_1_TO_2 / 100
            if not open_loop
            else tracked_measured_power[-1] * (1 + i * 0.01))  # continues drifting after open loop switch

        measured_data = measured_data.reset_index(drop=True)
        true_data = true_data.reset_index(drop=True)
        true_data.index = true_data.index.drop_duplicates()
        saved_slice = true_data.loc[INPUT_TIME:INPUT_TIME + STEP_SIZE - 1,
                      ['nfd-1-cps', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position',
                       'ss2-active-state', 'ss2-position', 'amdr-position', 'nfd-1-cr', 'Neutron Counts']
                      ]

        saved_slice.loc[:, 'controller belief'] = measured_data.loc[INPUT_TIME:INPUT_TIME + STEP_SIZE - 1,
                                                  'nfd-1-cps'] * CHANNEL_1_TO_2 / 100
        saved_slice.loc[:, 'Demand'] = demand
        #saved_slice.loc[:, 'MPC Prediction'] = mpc_pred
        saved_results = pd.concat([saved_results, saved_slice])
        #Uncomment this to print live results as the code runs
        #print('Demand: ',demand )
        #print('Power: ', true_data['Neutron Counts'].iloc[-1] / (counts_to_power_ratio * 100))
        #print('RR Position: ', true_data['rr-position'].iloc[-1])
        #print('##########################')

    #Re-adding the lists to the dataframe of saved results
    saved_results.reset_index(inplace=True, drop=True)
    saved_results['Demand'] = demand_curve[:len(saved_results)]
    saved_results['Power'] = saved_results['Neutron Counts'] / (counts_to_power_ratio * 100)
    saved_results['Time'] = np.arange(0, len(saved_results))
    saved_results['False Power'] = tracked_measured_power
    rmse, effort = calc_metrics(saved_results)
    #Calculating runtime average and standard deviation
    if cfg.method == 'MPC':
        avg_mpc = np.mean(np.array(mpc_runtimes))
        avg_mhe = np.mean(np.array(mhe_runtimes))
        std_mpc = np.std(np.array(mpc_runtimes))
        std_mhe = np.std(np.array(mhe_runtimes))
    else:
        avg_mpc = 0
        avg_mhe = 0
        std_mpc = 0
        std_mhe = 0
    std_total = np.std(np.array(total_runtimes))
    avg_total = np.mean(np.array(total_runtimes))
    #If the code is being called from another script, we stop the run here so that graphing does interrupt sensitivity analysis
    if cfg.return_metrics:
        return {'rmse': rmse, 'control_effort': effort, 'avg_mpc_runtime': avg_mpc, 'avg_mhe_runtime': avg_mhe, 'avg_total_runtime': avg_total,
                'std_mpc_runtime': std_mpc, 'std_mhe_runtime': std_mhe, 'std_total_runtime': std_total}
    #Saving csv results from the run
    saved_name = 'final_paper_runs/' + cfg.method + '_' + cfg.scenario + '_' + cfg.demand_scenario + '_seed3.csv'
    saved_results.to_csv(saved_name)

    #Building graph if running the code manually
    plt.rc('font', size=14)
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].plot(saved_results['Time'] / 60, saved_results['Power'], label='Power')
    ax[0].plot(saved_results['Time'] / 60, saved_results['Demand'], label='Demand')
    # ax[0].axvline(x=1000/60, color='red', linestyle='--', linewidth=1, label='Anomaly Start')
    # ax[0].axvline(x=detected*10 / 60, color='red', linestyle='-', linewidth=1, label='Anomaly Detected')
    ax[0].set_title(f'{cfg.method} Setpoint Tracking - {cfg.scenario}', fontsize=15)
    ax[0].set_ylim(0, 1.0)
    for i in range(len(starts)):
        ax[0].axvspan(starts[i] / 30, (starts[i] + lens[i]) / 30, color='red', alpha=0.3)
    ax[0].set_xlabel('Time (min)')
    ax[0].set_ylabel('Normalized Power')
    ax[0].legend()

    ax[1].plot(saved_results['Time'] / 60, saved_results['nfd-1-cr'], label='Change Rate')
    ax[1].set_title(f'{cfg.method} Change Rate - {cfg.scenario}')
    ax[1].set_xlabel('Time (min)')
    ax[1].set_ylabel('Change Rate')
    ax[1].legend()

    if rod == 'RR':
        rod_positions = saved_results['rr-position']
    if rod == 'AMDR':
        rod_positions = saved_results['amdr-position']
    if rod == 'SS2':
        rod_positions = saved_results['ss2-position']
    ax[2].plot(saved_results['Time'] / 60, rod_positions, label=f'{rod} Position')
    ax[2].set_title(f'{cfg.method} Rod Position - {cfg.scenario}')
    ax[2].set_xlabel('Time (min)')
    ax[2].set_ylabel('Rod Height (cm)')
    ax[2].legend()
    fig.subplots_adjust(wspace=0.233, hspace=0.171, left=0.055, bottom=0.14, right=0.988, top=0.93)
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = SimConfig()
    main_pke(cfg)

