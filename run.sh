#!/usr/bin/env bash
python MainExtrema.py -loss gan -lrg 3e-4 -lrd 3e-4 -gp_weight 100 -gp_center 0 -gp_inter 1
python MainExtrema.py -loss gan -lrg 3e-4 -lrd 3e-4 -gp_weight 100 -gp_center 0 -gp_inter 0
python MainExtrema.py -loss gan -lrg 3e-4 -lrd 3e-4 -gp_weight 0 -gp_center 0
python MainExtrema.py -loss gan -lrg 1e-4 -lrd 3e-4 -gp_weight 100 -gp_center 0
python MainExtrema.py -loss wgan -nd 5 -ng 1 -lrg 3e-4 -lrd 3e-4 -gp_weight 100 -gp_center 1
python MainExtrema.py -loss wgan -nd 5 -ng 1 -lrg 3e-4 -lrd 3e-4 -gp_weight 100 -gp_center 0
python MainExtrema.py -loss wgan -nd 5 -ng 1 -lrg 3e-4 -lrd 3e-4 -gp_weight 10 -gp_center 0
python MainExtrema.py -loss wgan -nd 5 -ng 1 -lrg 3e-4 -lrd 3e-4 -gp_weight 100 -gp_center 0 -gp_inter 1
python MainExtrema.py -loss wgan -nd 5 -ng 1 -lrg 3e-4 -lrd 3e-4 -gp_weight 10 -gp_center 0 -gp_inter 1

