
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