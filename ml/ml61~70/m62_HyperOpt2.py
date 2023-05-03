# 최소값 찾는 것
# 베이지안옵티마이제이션은 최대값을 찾는 것
# pip install hyperopt
import hyperopt
print(hyperopt.__version__)     # 0.2.7
import numpy as np

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),
    'x2' : hp.quniform('x2', -15, 15, 1)
    #      hp.quniform(label, low, high, q)
}
print(search_space)

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 - 20*x2
    
    return return_value
    # 권장리턴방식 return {'loss':return_value, 'status':"STATUS_OK"}

trial_val = Trials()

best = fmin(
    fn = objective_func,
    space = search_space,
    algo = tpe.suggest,         # 디폴트
    max_evals = 20,  # ?번 돌리겠다
    trials = trial_val,
    rstate = np.random.default_rng(seed= 10)
)

print('best : ', best)
# best :  {'x1': 0.0, 'x2': 15.0}

# print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, ..., {'loss': 0.0, 'status': 'ok'}]

# print(trial_val.vals)
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, ...], 
# 'x2': [11.0, 10.0, -4.0, -5.0, ...]}

##### pandas 데이터프레임에 trial_val, vals를 넣어봐라 #####
import pandas as pd

results = [aaa['loss'] for aaa in trial_val.results]     # trial_val 결과값들을 반복해라 반복하는 걸 aaa라고 라고 그거에 대한 결과를 aaa['loss']에 저장해라

# for aaa in trial_val.results:
#     losses.append(aaa['loss'])        # 위 한 줄과 동일

df = pd.DataFrame({'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2'],
                   'results' : results})
print(df)

    




