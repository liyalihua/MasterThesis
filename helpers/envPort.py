import gym
from gym.utils import seeding
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List
matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame, input data
        stock_dim : int, number of unique stocks
        initial_amount : int, start capital
        state_space: int, the dimension of input features
        action_space: int, equals stock dimension
        day: int, an increment number to control date
        tech_indicator_list: list, a list of technical indicator names in the dataframe
        transaction_cost_pct: float, transaction cost percentage per trade
        reward_scaling: float, scaling factor for reward, good for training

    Methods
    -------
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                initial_amount,
                transaction_cost_pct,
                state_space,
                action_space,
                lookback=252,
                day = 0):
        super(gym.Env, self).__init__() 
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim  
        self.initial_amount = initial_amount
        self.state_space = state_space
        self.action_space = action_space
        self.terminal = False  
        
        # action space is the number of stock
        self.action_space = spaces.Box(low = 0, high = 10,shape = (self.action_space,)) 
        # observation space is the daily return matrix with one year lookback period
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,self.stock_dim+1))

        # select remaining data for a given date, and select the return matrix for that day. that will be the state, if we don't add more
        self.data = self.df.loc[self.day,:]
        self.ret = self.data.iloc[0]['ar']
        self.state = self.ret
   
        # book keeper
        # date memory
        self.date_memory=[self.data.date.unique()[0]]
        self.portfolio_value = self.initial_amount
        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [1]
        # self.portfolio_return_memory = [0]
        # eq initialization, could be random if needed.
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]

        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()
            
            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe Ratio: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,{}

        else:
            print("Model actions: ",actions)
            weights = self.softmax_normalization(actions) 
            print("Weights: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.ret = self.data.iloc[0]['ar']
            self.state = self.ret
            
            # debug print
            print(self.day)
            print(self.state)
            # print(f"last day memory: {last_day_memory}")
            # print(f"counter: {self.day}")
            # print(f"current day info: {self.data}")

            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            print(f'portfolio return:{portfolio_return}')
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            print(f'portfolio value:{self.portfolio_value}')
            self.portfolio_value = new_portfolio_value
            print(f"new portfolio value: {new_portfolio_value}")

            #debug print
            # print(f"close value t= {self.day} : {self.data.close.values}")
            # print(f"close value t= {self.day-1} : {last_day_memory.close.values}")
            # print(f" .......")
            # print(f"actions: {actions}")
            # print(f"weights: {weights}")
            # print(f" .......")
            # print(f"portfolio return Type: {type(portfolio_return)}")
            # print(f"portfolio return: {portfolio_return}")
            # print(f"new portfolio value Type: {type(new_portfolio_value)}")
            # print(f"new portfolio value: {new_portfolio_value}")

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value 
            self.reward = new_portfolio_value 
            #print("Step reward: ", self.reward)
            # we change the reward to log portfolio return, so the reward will be additive
            # self.reward = np.round(np.log(1+portfolio_return-0),4)

        return self.state, self.reward, self.terminal, {'weights':weights}

    def reset(self):
        # date and state
        self.terminal = False 
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.ret = self.data.iloc[0]['ar']
        self.state =self.ret
        print(f'Environment reset. Day: {self.day}, State: {self.state}')

        # bookkeeping
        self.date_memory=[self.data.date.unique()[0]] 
        self.portfolio_value = self.initial_amount
        # self.portfolio_return_memory = [0]
        self.portfolio_return_memory = [1] # using port return
        self.asset_memory = [self.initial_amount]

        #initialize action equally
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        # self.actions_memory=[[1/(self.stock_dim-1)]*(self.stock_dim-1)] # substract 1 as we have day progress bar along the way
        #random initialization at rest
        # ls= np.random.default_rng(seed=0).random((42))
        # self.actions_memory = [(ls/ls.sum()).tolist()]
        # Another test
        # self.actions_memory = initWeight
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def close(self):
      pass

