from qutip.qip import QubitCircuit, gate_sequence_product, qubit_states
from qutip import Qobj, gate_expand_1toN, gate_expand_2toN, csign, cnot, rx, ry, rz
import numpy as np

def h(N=None, target=0):

    if N is not None:
        return gate_expand_1toN(h(), N, target)
    else:
        return Qobj(np.array([[1.0, 1.0], [1.0, -1.0]], dtype='complex128') / np.sqrt(2.0), dims=[[2],[2]])

def x(N=None, target=0):

    if N is not None:
        return gate_expand_1toN(x(), N, target)
    else:
        return Qobj(np.array([[0., 1.0], [1.0, 0.]], dtype='complex128'), dims=[[2],[2]])

def y(N=None, target=0):

    if N is not None:
        return gate_expand_1toN(y(), N, target)
    else:
        return Qobj(np.array([[0., -1j], [1j, 0.]], dtype='complex128'), dims=[[2],[2]])

def z(N=None, target=0):
    if N is not None:
        return gate_expand_1toN(z(), N, target)
    else:
        return Qobj(np.array([[1., 0.0], [0.0, -1.]], dtype='complex128'), dims=[[2],[2]])

def u3(theta3, N=None, target=0):
    if N is not None:
        return gate_expand_1toN(u3(theta3), N, target)

    else:
        mat = np.array([[np.cos(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    -np.sin(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)],
                   [np.sin(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    np.cos(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)]], dtype='complex128')

        return Qobj(mat, dims=[[2],[2]])

def cy(N=None, control=0, target=1):

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cy(), N, control, target)
    else:
        return Qobj([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, -1.j],
                     [0, 0, 1.j, 0]],
                    dims=[[2, 2], [2, 2]])

def crx(theta, N=None, control=0, target=1):

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(crx(theta), N, control, target)
    else:
        return Qobj([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, np.cos(theta / 2), 1j * np.sin(theta/2)],
                     [0, 0, 1j * np.sin(theta / 2), np.cos(theta / 2)]],
                    dims=[[2, 2], [2, 2]])

def cry(theta, N=None, control=0, target=1):

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cry(theta), N, control, target)
    else:
        return Qobj([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, np.cos(theta / 2), -np.sin(theta/2)],
                     [0, 0, np.sin(theta / 2), np.cos(theta / 2)]],
                    dims=[[2, 2], [2, 2]])

def crz(theta, N=None, control=0, target=1):

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(crz(theta), N, control, target)
    else:
        return Qobj([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, np.exp(1j * theta / 2), 0],
                     [0, 0, 0, np.exp(-1j * theta / 2)]],
                    dims=[[2, 2], [2, 2]])

def cu3(theta3, N=None, control=0, target=1):

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cu3(theta3), N, control, target)
    else:
        return Qobj([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, np.cos(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    -np.sin(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)],
                   [0, 0, np.sin(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    np.cos(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)]],
                    dims=[[2, 2], [2, 2]])


class QCircuit:
    def __init__(self, qubits):
        self.state=qubit_states(qubits)
        self.qubits = qubits

    def H(self, target):
        self.state = h(self.qubits, target=self.qubits - 1 - target) * self.state

    def X(self, target):
        self.state = x(self.qubits, target=self.qubits - 1 - target) * self.state

    def Y(self, target):
        self.state = y(self.qubits, target=self.qubits - 1 - target) * self.state

    def Z(self, target):
        self.state = z(self.qubits, target=self.qubits - 1 - target) * self.state

    def RX(self, target, theta):
        self.state = rx(theta, self.qubits, self.qubits - 1 - target) * self.state

    def RY(self, target, theta):
        self.state = ry(theta, self.qubits, self.qubits - 1 - target) * self.state

    def RZ(self, target, theta):
        self.state = rz(theta, self.qubits, self.qubits - 1 - target) * self.state

    def U3(self, target, theta3):
        self.state = u3(theta3, self.qubits, self.qubits - 1 - target) * self.state

    def CX(self, control, target):
        self.state = cnot(self.qubits, control, self.qubits - 1 - target) * self.state

    def CY(self, control, target):
        self.state = cy(self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def CZ(self, control, target):
        self.state = csign(self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def CRX(self, control, target, theta):
        self.state = crx(theta, self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def CRY(self, control, target, theta):
        self.state = cry(theta, self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def CRZ(self, control, target, theta):
        self.state = crz(theta, self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def CU3(self, control, target, theta3):
        self.state = cu3(theta3, self.qubits, self.qubits - 1 - control, self.qubits - 1 - target) * self.state

    def get_state(self, flatten=True):
        if flatten:
            return self.state[:]
        else:
            return self.state[:].reshape((self.state.dims[0]))

    def measure_list(self, measured_qubits):
        measured_qubits.sort()
        measured_qubits.reverse()
        sum_list = list(range(self.qubits))
        for q in measured_qubits:
            sum_list.remove(self.qubits - 1 - q)

        return sum_list

    def update_state(self, new_state):
        new_state.reshape((2,)*self.qubits)
        self.state = Qobj(new_state, dims=self.state.dims)

    def reset(self):
        self.state = qubit_states(self.qubits)

    def measure_probs(self, probabilities, n_samples):
        measurements = np.zeros_like(probabilities).flatten()
        cum_probs = np.cumsum(probabilities)
        p = np.random.rand(n_samples)
        pos = np.searchsorted(cum_probs, p)
        for pos_i in pos:
            measurements[pos_i] += 1
        measurements.reshape(probabilities.shape)
        measurements /= np.sum(measurements)

        return measurements

    def get_all_probabilities(self):
        probabilities = np.abs(self.state[:].reshape(self.state.dims[0])) ** 2
        return probabilities

    def measure_all(self, n_samples, return_probabilities = False):
        # Esto puede optimizarse utilizando tensornetworks??
        # probabilities = tn.Node(np.abs(self.state.tensor) ** 2)
        probabilities = self.get_all_probabilities()
        measurements = self.measure_probs(probabilities, n_samples).reshape(probabilities.shape)

        if return_probabilities:
            return measurements, probabilities

        else:
            return measurements

    def get_some_probabilities(self, measured_qubits):
        probabilities = self.get_all_probabilities()
        l = tuple(self.measure_list(measured_qubits))
        con_probabilities = np.sum(probabilities, axis=l)

        return con_probabilities


    def measure_some_qubits(self, n_samples, measured_qubits, return_probabilities = False):
        con_probabilities = self.get_some_probabilities(measured_qubits)
        measurements = self.measure_probs(con_probabilities, n_samples).reshape(con_probabilities.shape)

        if return_probabilities == False:
            return measurements

        else:
            return measurements, con_probabilities



class GeneralVariationalAnsatz(QCircuit):
    def __init__(self, qubits, layers):
        QCircuit.__init__(self, qubits, user_gates={"U3": u3})
        self.layers = layers
        self.parameters = np.random.rand(2 * layers + 1, qubits, 3)
        self.qubits = qubits

    def reset(self):
        self.state = np.zeros_like(self.state)
        self.state[(0,) * self.qubits] = 1.

    def entangling_layer_1(self, layer):
        for q in range(self.qubits):
            lab = r'\theta_{2 * %s, %s}' % (layer, q)
            self.add_gate("U3", targets=self.qubits - 1 - q, arg_value=self.parameters[2 * layer, q],
                 arg_label=lab)

        for q in range(0, self.qubits, 2):
            self.add_gate("CSIGN", controls=self.qubits - 1 - q, targets=((self.qubits - q - 2) % self.qubits))

    def entangling_layer_2(self, layer):
        for q in range(self.qubits):
            lab = r'\theta_{2 * %s, %s}' % (layer, q)
            self.add_gate("U3", targets=self.qubits - 1 - q, arg_value=self.parameters[2 * layer + 1, q],
                          arg_label=lab)

        for q in range(1, self.qubits, 2):
            self.add_gate("CSIGN", controls=self.qubits - 1 - q, targets=(self.qubits - q - 2) % self.qubits)  # Coge Z?

    def final_layer(self):
        for q in range(self.qubits):
            lab = r'\theta_{2 * %s, %s}' % (-1, q)
            self.add_gate("U3", targets=self.qubits - 1 - q, arg_value=self.parameters[-1, q],
                          arg_label=lab)

    def mult_matrices(self):
        for l in range(self.layers):
            self.entangling_layer_1(l)
            self.entangling_layer_2(l)

        self.final_layer()

    def run(self):
        self.mult_matrices()
        props = self.propagators()
        U1 = gate_sequence_product(props)
        self.state= U1 * qubit_states(N=self.qubits)

    def update_parameters(self, new_parameters):
        if self.parameters.shape != new_parameters.shape:
            raise ValueError("Shape is not appropiate")
        self.parameters = new_parameters


class GeneralVariationalAnsatz_2(QubitCircuit):
    def __init__(self, qubits, layers):
        QubitCircuit.__init__(self, qubits, user_gates={"U3": u3})
        self.layers = layers
        self.parameters = np.random.rand(2 * layers + 1, qubits, 3)
        self.qubits = qubits
        self.state=qubit_states(N=self.qubits)

    def reset(self):
        self.state=qubit_states(N=self.qubits)

    def entangling_layer_1(self, layer):
        for q in range(self.qubits):
            self.state = u3(self.parameters[2 * layer, q], N=self.qubits, target=self.qubits - 1 - q) * self.state

        for q in range(0, self.qubits, 2):
            self.state = csign(self.qubits, self.qubits - 1 - q, ((self.qubits - q - 2) % self.qubits)) * self.state

    def entangling_layer_2(self, layer):
        for q in range(self.qubits):
            self.state = u3(self.parameters[2 * layer + 1, q], N=self.qubits, target=self.qubits - 1 - q) * self.state

        for q in range(1, self.qubits, 2):
            self.state = csign(self.qubits, self.qubits - 1 - q, ((self.qubits - q - 2) % self.qubits)) * self.state

    def final_layer(self):
        for q in range(self.qubits):
            self.state = u3(self.parameters[-1, q], N=self.qubits, target=self.qubits - 1 - q) * self.state

    def run(self):
        self.reset()
        for l in range(self.layers):
            self.entangling_layer_1(l)
            self.entangling_layer_2(l)

        self.final_layer()

    def update_parameters(self, new_parameters):
        if self.parameters.shape != new_parameters.shape:
            raise ValueError("Shape is not appropiate")
        self.parameters = new_parameters



