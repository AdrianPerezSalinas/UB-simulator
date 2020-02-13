import numpy as np
from scipy.optimize import minimize

h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype='complex128') / np.sqrt(2.0)
x = np.array([[0., 1.0], [1.0, 0.]], dtype='complex128')
z = np.array([[1., 0.0], [0.0, -1.]], dtype='complex128')
y = np.array([[0., -1j], [1j, 0.]], dtype='complex128')

cx = np.zeros((2, 2, 2, 2), dtype='complex128')
cx[0, 0] = np.eye(2)
cx[1, 1] = x

cy = np.zeros((2, 2, 2, 2), dtype='complex128')
cy[0, 0] = np.eye(2)
cy[1, 1] = y

cz = np.zeros((2, 2, 2, 2), dtype='complex128')
cz[0, 0] = np.eye(2)
cz[1, 1] = z


def rx(theta):
    r = np.array([[np.cos(theta / 2), 1j*np.sin(theta / 2)], [1j*np.sin(theta / 2), np.cos(theta / 2)]], dtype='complex128')
    return r

def ry(theta):
    r = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype='complex128')
    return r

def rz(theta):
    r = np.array([[np.exp(1j * theta / 2), 0], [0, np.exp(-1j * theta / 2)]], dtype='complex128')
    return r

def u3(theta3):
    r = np.array([[np.cos(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    -np.sin(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)],
                   [np.sin(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    np.cos(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)]], dtype='complex128')

    return r


def crx(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = rx(theta)
    return c


def cry(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = ry(theta)

    return c


def crz(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = rz(theta)

    return c


def cu3(theta3):
    c = np.zeros((2, 2, 2, 2))
    c[0, 0] = np.eye(2)
    c[1, 1] = u3(theta3)

    return c


def rst(qubits):
    tens = np.array([1., 0.], dtype='complex128')
    for _ in range(qubits - 1):
        tens = np.tensordot(tens, np.array([1., 0.], dtype='complex128'), axes=0)

    return tens

class QCircuit:
    def __init__(self, qubits):
        self.qubits = qubits
        self.state = rst(self.qubits)
        self.state_shape = self.state.shape

    def reset(self):
        self.state = rst(self.qubits)

    def select_op_1(self, target):
        return [-self.qubits + target, 1]

    def select_target_1(self, target):
        l = list(range(-1, -self.qubits - 1, -1))
        l[self.qubits - target - 1] = 1
        return l

    def select_op_2(self, control, target):
        return [-self.qubits + control, 1, -self.qubits + target, 2]

    def select_target_2(self, control, target):
        l = list(range(-1, -self.qubits - 1, -1))
        l[self.qubits - control - 1] = 1
        l[self.qubits - target - 1] = 2
        return l

    def measure_list(self, measured_qubits):
        measured_qubits.sort()
        measured_qubits.reverse()
        sum_list = list(range(self.qubits))
        for q in measured_qubits:
            sum_list.remove(self.qubits - 1 - q)

        return sum_list


    def H(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([h, self.state], [l1, l2])

    def X(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([x, self.state], [l1, l2])

    def Y(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([y, self.state], [l1, l2])

    def Z(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([z, self.state], [l1, l2])

    def RX(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([rx(theta), self.state], [l1, l2])

    def RY(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([ry(theta), self.state], [l1, l2])

    def RZ(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([rz(theta), self.state], [l1, l2])

    def U3(self, target, theta3):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.state = tn.ncon([u3(theta3), self.state], [l1, l2])

    def CX(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.state = tn.ncon([cx, self.state], [l1, l2])

    def CY(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.state = tn.ncon([cy, self.state], [l1, l2])

    def CZ(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.state = tn.ncon([cz, self.state], [l1, l2])

    def CRX(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.state = tn.ncon([crx(theta), self.state], [l1, l2])

    def CRY(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.state = tn.ncon([cry(theta), self.state], [l1, l2])

    def CRZ(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.state = tn.ncon([crz(theta), self.state], [l1, l2])

    def CU3(self, control, target, theta3):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.state = tn.ncon([cu3(theta3), self.state], [l1, l2])

    def get_state(self, flatten=True):
        if flatten:
            return self.state.flatten()
        else:
            return self.state


    def update_state(self, tensor):
        if tensor.shape != self.state_shape:
            raise IndexError('Dimension mismatch')

        self.state = tensor

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
        probabilities = np.abs(self.state) ** 2
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
        # probabilities = np.abs(self.state.tensor) ** 2
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
        self.layers = layers
        QCircuit.__init__(self, qubits)
        self.parameters = np.random.rand(2*layers + 1, qubits, 3)

    def entangling_layer_1(self, layer):
        for q in range(self.qubits):
            self.U3(q, self.parameters[2 * layer, q])

        for q in range(0, self.qubits, 2):
            self.CZ(q, (q + 1)%self.qubits)

    def entangling_layer_2(self, layer):
        for q in range(self.qubits):
            self.U3(q, self.parameters[2 * layer + 1, q])

        for q in range(1, self.qubits, 2):
            self.CZ(q, (q + 1)%self.qubits)

    def final_layer(self):
        for q in range(self.qubits):
            self.U3(q, self.parameters[-1, q])

    def run(self):
        self.reset()
        for l in range(self.layers):
            self.entangling_layer_1(l)
            self.entangling_layer_2(l)

        self.final_layer()

    def update_parameters(self, new_parameters):
        if self.parameters.shape != new_parameters.shape:
            raise ValueError ("Shape is not appropiate")
        self.parameters = new_parameters


class one_qubit_approximant(QCircuit):
    def __init__(self, layers, domain, f):
        QCircuit.__init__(self, 1)
        self.layers = layers
        self.Domain = domain
        self.Domain_tensor = np.array([np.diag(np.array([x_i, 1, 1])) for x_i in domain])
        self.F = f
        self.params_shape = (layers, 3)
        self.params = np.zeros(self.params_shape)
        self.params_x = tn.ncon([self.params, self.Domain_tensor], [[-1, 1],[-2, 1, -3]])
        #El cálculo grande parámetros * x debe hacerse aquí y sólo una vez

    def layer(self, l, point):
        theta = self.params_x.tensor[l, point] # Se puede hacer con tensorflow, pero desde el principio
        self.RY(0, theta[0])
        self.RY(0, theta[1])
        self.RZ(0, theta[2])


    def run(self):
        y = np.empty_like(self.Domain)
        for i, x_i in enumerate(self.Domain):
            self.reset()
            for l in range(self.layers):
                self.layer(l, i)

            y[i] = self.get_all_probabilities()[1]

        return y


    def chi_square(self):
        y = self.run()
        return np.mean((y - self.F(self.Domain))**2) * 0.5


    def update_parameters(self, new_parameters):
        self.params = new_parameters
        self.params_x = tn.ncon([self.params, self.Domain_tensor], [[-1, 1], [-2, 1, -3]])

    def minimizing_function(self, parameters):
        parameters = parameters.reshape(self.params_shape)
        self.update_parameters(parameters)
        current_value = self.chi_square()
        return current_value

    def get_optimal_parameters(self):
        init_parameters = 2 * (np.random.rand(self.params_shape[0] * self.params_shape[1]) - .5)

        result = minimize(self.minimizing_function,init_parameters)
        return result
