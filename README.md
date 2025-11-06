# Implementation of the algorithms from "Differentially Private Policy Gradient", available as a pre-print at https://arxiv.org/abs/2402.05525.

The implementation of DPPG (discrete and continuous) is based on the cleanRL implementation of PPO.

First install the requirements (same as ```cleanrl```):
```pip install -r requirements.txt```

The codebase contains 4 files to reproduce the experiments from the paper:
- ```dppg_discrete.py```: code to reproduce the experiments for discrete DPPG (CartPole, Acrobot)
- ```dppg_continuous.py```: code to reproduce the experiments for continuous DPPG (MuJoCo, Dosing)
- ```dppg_tabular.py```: code to reproduce the experiments for tabular/linear MDPs (Riverswim)
- ```ucbvi_tabular.py```: code to reproduce the baseline results (UCBVI) for tabular/linear MDPs (Riverswim), from the official implementation https://github.com/XingyuZhou989/PrivateTabularRL

To run the experiments, use for instance the following command:
```python dppg_discrete.py --env_id $ENV_ID --args```

The relevant hyperparameters can be found in the paper.

