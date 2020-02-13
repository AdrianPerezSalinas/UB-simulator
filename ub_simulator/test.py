import copy
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi
from time import time

from qutip.qip.device import Processor
from qutip.operators import sigmaz, sigmay, sigmax, destroy
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.qip.operations import rx, ry, rz, hadamard_transform
from qutip.qip.circuit import qubit_states

processor = Processor(N=1)
state = qubit_states()
processor.add_control(sigmay(), targets=0, label="sigmay")
processor.add_control(sigmaz(), targets=0, label="sigmaz")
'''for pulse in processor.pulses:
    pulse.print_info()'''



processor.pulses[0].coeff = np.array([.5])
processor.pulses[0].tlist = np.array([0., pi])
start = time()
state = processor.run_state(init_state=state)
for i in range(100):
    processor.run_state(init_state=state.states[-1])
end = time()
print(end - start)

start = time()
state = qubit_states()
processor.pulses[0].coeff = np.ones(100) * 0.5
processor.pulses[0].tlist = np.arange(0, 100*np.pi, np.pi) # Esto es MUCHÍSIMO MÁS RÁPIDO
processor.run_state(init_state=state)
end = time()
print(end - start)