from mesa import Agent
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
from model import *
import numpy as np
import random

class CorruptionAgent(Agent):
    """ Agents in the model."""
    def __init__(self, unique_id, model, parent=None):
        """Create an instance of agents with properties:
        steps = 0, counter so that they can die
        parent = parent, if it was not spawned in the beginning
        b = risk-aversion, taken from a uniform distribution, according to the paper
        k = capital endowment, initially taken from a uniform distribution,
        then inherited from parent's income and taken from distribution."""
        super().__init__(unique_id, model)
        self.steps = 0 
        self.id = unique_id
        self.parent = parent
        self.get_risk_aversion()
        self.get_capital_endowment()
        #self.p=0
      
    def get_risk_aversion(self):
        self.b = np.random.uniform(self.model.b_bar-self.model.b_range/2,
                                   self.model.b_bar+self.model.b_range/2)
        
    def get_capital_endowment(self):
        if self.parent != None:
            self.k = (self.model.k_min + ((self.parent.y -self.model.y_min)/
                      (self.model.y_max - self.model.y_min)) *
                    (self.model.k_max-self.model.k_min) * (
                        1-self.model.theta) + 
                      (self.model.theta*np.random.uniform(
                          self.model.k_min, self.model.k_max)))
        else: 
            self.k = np.random.uniform(
                self.model.k_bar - self.model.k_range/2, 
                self.model.k_bar + self.model.k_range/2)

          
    def spawn_new_agent(self):
        """Spawn new agent with the spawning agent as parent, add to model."""
        new_agent=CorruptionAgent(self.unique_id, model, parent = self)
        self.model.schedule.add(new_agent)
        
             
    
    def choose_dishonesty_level(self):  
        """Choose a dishonesty level to maximize utility."""
        self.p = 1/(2*self.b*self.model.gamma**2*(
            1-self.model.q)*self.model.S)
        
    def get_income(self):
        """Get income y:
        One part is fixed, due to non-illicit corrupt work, 
        the other part due to corrupt work."""
        self.y = (1-self.p)*(1-self.model.q)*self.model.S*self.k + (
        self.p * np.random.normal(self.model.S*self.k, 
                                  (1-self.model.q)*self.model.gamma * 
                              self.model.S * self.k))
    def corrupt(self):
        ''' Choose dishonesty level, and associated income, add to steps.'''
        #check if
        self.choose_dishonesty_level()
        self.get_income()
        self.steps += 1
        
    def procreate(self):
        """ Breed new agent, then die."""
        self.spawn_new_agent()
        if self.steps >= 1:
            self.model.schedule.remove(self)