from collections import defaultdict
import math
import operator
from palmetto import Palmetto
import torch
import numpy as np
dataset = [0.1117, 0.1112, 0.1133, 0.1117, 0.1125, 0.1103, 0.1099, 0.1099, 0.1095]
y_max = 10.0
y_min = 0.0
dataset = (y_max-y_min)*(dataset-np.min(dataset))/(np.max(dataset)-np.min(dataset))
print(dataset)


dataset = ['5.78947368','4.47368421','10','5.78947368','7.89473684','2.10526316','1.05263158','1.05263158','0']
palmetto = Palmetto()
# cp = palmetto.get_coherence(dataset, coherence_type="cp")
# print('cp',cp)
# ca = palmetto.get_coherence(dataset, coherence_type="ca")
# print('ca',ca)
# npmi = palmetto.get_coherence(dataset, coherence_type="npmi")
# print('npmi',npmi)
uci = palmetto.get_coherence(dataset, coherence_type="uci")
print('uci',uci)
