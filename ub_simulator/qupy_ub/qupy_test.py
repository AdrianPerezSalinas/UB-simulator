from qubit import Qubits
from qubit_error import Qubits_error
import qubit as q
import numpy as np
from qupy.operator import X, Z, H



a = Qubits_error(2)
a.perfect_gate(H, 1)
print(a.state.reshape((2**2, 2**2)))
a.controlled_gate(X, 0, 1)
print(a.state.reshape((2**2, 2**2)))
