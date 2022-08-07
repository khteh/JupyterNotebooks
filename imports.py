# Be sure to include this magic before you import Matplotlib, as it may not work if you do not
%matplotlib inline
import sys
!{sys.executable} -m pip install --user numpy pandas matplotlib seaborn
import numpy, pandas, matplotlib, seaborn
import matplotlib.pyplot as pyplot
seaborn.set(style="darkgrid")
