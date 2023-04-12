import pandas as pd
import numpy as np
from scipy.stats import norm


chat_id = 252623629

def solution(x_success: int,
x_cnt: int,
y_success: int,
y_cnt: int) -> bool:
control_conv = x_success / x_cnt
test_conv = y_success / y_cnt
dev = test_conv - control_conv
se = np.sqrt((test_conv*(1-test_conv)/y_cnt)+(control_conv*(1-control_conv)/x_cnt))
z_score = dev / se
p_val = 2*(1-norm.cdf(abs(z_score)))
if p_val < 0.03:
return True
else:
return False
