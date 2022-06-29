
# BBFootprint

## Installation
No particular installation is required, other than copying the `twiss/survey` as pickle files from
https://cernbox.cern.ch/index.php/s/bQqj58FQDVgGUfw


## About

The main tools of this backage are found in `Backend/Detuning.py` and `Backend/InteractionPoint.py`. One can extract all the relevant HO and LR information for each IPs using:
```python
import pandas as pd
import Backend.InteractionPoint as inp

# Importing twiss/survey files
twiss_b1  = pd.read_pickle('LHC_sequence/lhcb1_opticsfile30_twiss.pkl')
survey_b1 = pd.read_pickle('LHC_sequence/lhcb1_opticsfile30_survey.pkl')

twiss_b2  = pd.read_pickle('LHC_sequence/lhcb2_opticsfile30_twiss.pkl')
survey_b2 = pd.read_pickle('LHC_sequence/lhcb2_opticsfile30_survey.pkl')

# Creating beam objects:    
B1 = inp.Beam('b1',twiss_b1,survey_b1,
          Nb       = 1.15e11,
          E        = 6.8e12,
          emittx_n = 2.5e-6,
          emitty_n = 2.5e-6,
          dp_p0    = 0)
    
B2 = inp.Beam('b2',twiss_b2,survey_b2,
              Nb       = 1.15e11,
              E        = 6.8e12,
              emittx_n = 2.5e-6,
              emitty_n = 2.5e-6,
              dp_p0    = 0)

# Attaching beams to the IPs and extracting the relevant twiss/survey for the IP
IP1 = inp.InteractionPoint('ip1',B1,B2)
IP5 = inp.InteractionPoint('ip5',B1,B2)
```

From there, `IP1.b1.twiss` contains all the information necessary, including the coordinates in the lab frame, and `IP1.bb/IP1.lr/IP1.ho` contains dataframes with the BB parameters computed for `All BB/LR only/HO only`.

An example of footprint computation is found in `demo_detuning.ipynb`.

## Detuning

The detuning from a given LR (or HO) interaction can be computed analytically following:

```python
import Backend.Detuning as dtune

_bb = IP5.lr.loc['bb_lr.r5b1_13']

DQx,DQy = dtune.DQx_DQy( ax   = coord['x_n'],
                         ay   = coord['y_n'],
                         r    = _bb['r'],
                         dx_n = _bb['dx_n'],
                         dy_n = _bb['dy_n'],
                         A_w_s= _bb['A_w_s'],
                         B_w_s= _bb['B_w_s'],
                         xi   = IP5.b2.xi)
```

The explanation for the physics and the underlying equations can be found from the papers in:
https://cernbox.cern.ch/index.php/s/bQqj58FQDVgGUfw

Note that `A_w_s` and `B_w_s` are described in the addendum. These numbers are necessary for the general solution of the detuning, where the optics of the weak beam and the strong beam are not necessarily antisymmetric.