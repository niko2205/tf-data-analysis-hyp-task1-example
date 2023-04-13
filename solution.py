import pandas as pd
import numpy as np
from scipy.stats import norm


chat_id = 252623629

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool: 
  p1 = x_success / x_cnt 
  p2 = y_success / y_cnt 
  p = (x_success + y_success) / (x_cnt + y_cnt) 
  sigma = np.sqrt(p*(1-p)*(1/x_cnt+1/y_cnt)) 
  z = (p1 - p2) / sigma 
  p_value = 2 * (1 - norm.cdf(abs(z))) 
  return p_value < 0.03
