from mesa import Agent, Model
from mesa.time import StagedActivation
import numpy as np
from mesa.datacollection import DataCollector
import random

def total_capital(model):
    """ Determine total capital of the model as sum of all the agents' capitals"""
    agent_capitals = [a.k for a in model.schedule.agents]
    capital = np.sum(agent_capitals)
    return capital

def corruption_index(model):
    """Determine total corruption index of the model. 
    Weighted average of all the dishonesty levels,
    with weights being each agent's capital"""
    capital = total_capital(model)
    corruption_index = 1/capital * (
        np.sum([a.k*a.p for a in model.schedule.agents]))
    return corruption_index
    
def min_max_income(model):
    """ Determine minimum and maximum income level each generation."""
    min_income = np.min([a.y for a in model.schedule.agents])
    max_income = np.max([a.y for a in model.schedule.agents])
    return min_income, max_income

def social_capital(model):
    """Determine social capital each generation."""
    capital = total_capital(model)
    social_capital = model.alpha * capital
    return social_capital

def national_income(model):
    """Determine the national income as sum over all the individual incomes."""
    incomes = [a.y for a in model.schedule.agents]
    national_income = np.sum(incomes)
    return national_income

class CorruptionModel(Model):
    """A model with population agents."""
    def __init__(self, b_bar = 3, b_range = 1, 
                 alpha= 0.5,  gamma = 0.5, theta= 0.1, q_start= 0.1, 
                 population=1000, k_bar = 0.5, 
                 k_range = 1):
        """Create the model with the following parameters:
        Average level of risk aversion = b_bar
        Range of risk aversion = b_range
        Proportion of income spent on vigilance = gamma
        Mean human capital endowment in first generation = k_bar
        Equality in access to human capital = theta
        Initial value of social corruption = q_start
        Population = population
        Number of generations = generations
        Range of human capital endowment: k_range"""
        #Set parameters
        self.running = True
        self.num_agents= population
        self.b_bar = b_bar
        self.b_range = b_range
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.q = q_start
        self.k_bar = k_bar
        self.k_range = k_range
        self.k_min = k_bar -0.5*k_range
        self.k_max = k_bar +0.5*k_range
        self.S = alpha * (k_bar * population)
        self.schedule = StagedActivation(self, stage_list=["corrupt", 
                                                           "procreate"])
        self.running = True
    
    
        #Create agents
        for i in range(self.num_agents):
            a = CorruptionAgent(i, self)
            self.schedule.add(a)       
              
        #Add data to report
        self.datacollector = DataCollector(
            model_reporters={"Total capital": total_capital, 
                             "Corruption Index": corruption_index, 
                            "National Income": national_income},
            agent_reporters={"Dishonesty": lambda a:a.p})
        
        
        
    def step(self):
        ''' Advance the model by one step. Do this in two stages:
        In first stage, corrupt, then report data and update model parameters.
        In second stage, breed and die.'''
        for stage in self.schedule.stage_list:
            for agent in self.schedule.agents[:]:
                getattr(agent, stage)() 
            if stage == "corrupt":
                self.q = corruption_index(self)
                self.datacollector.collect(self)
                self.y_min, self.y_max = min_max_income(self)
                #self.k_min, self.k_max = min_max_capital(self)
            self.K = total_capital(self)
            self.S = social_capital(self)
    
                
    