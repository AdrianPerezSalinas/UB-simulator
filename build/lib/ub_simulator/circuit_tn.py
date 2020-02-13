import tensornetwork as tn
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

h = tn.Node(np.array([[1.0, 1.0], [1.0, -1.0]], dtype='complex128') / np.sqrt(2.0), name='Hadamard')
x = tn.Node(np.array([[0., 1.0], [1.0, 0.]], dtype='complex128'), name='sigma_x')
z = tn.Node(np.array([[1., 0.0], [0.0, -1.]], dtype='complex128'), name='sigma_z')
y = tn.Node(np.array([[0., -1j], [1j, 0.]], dtype='complex128'), name='sigma_y')

cx = np.zeros((2, 2, 2, 2), dtype='complex128')
cx[0, 0] = np.eye(2)
cx[1, 1] = x.tensor
cx = tn.Node(cx, name='CX')

cy = np.zeros((2, 2, 2, 2), dtype='complex128')
cy[0, 0] = np.eye(2)
cy[1, 1] = y.tensor
cy = tn.Node(cy, name='CY')

cz = np.zeros((2, 2, 2, 2), dtype='complex128')
cz[0, 0] = np.eye(2)
cz[1, 1] = z.tensor
cz = tn.Node(cz, name='CZ')


def rx(theta):
    r = np.array([[np.cos(theta / 2), 1j*np.sin(theta / 2)], [1j*np.sin(theta / 2), np.cos(theta / 2)]], dtype='complex128')
    r = tn.Node(r, name='RX')
    return r

def ry(theta):
    r = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype='complex128')
    r = tn.Node(r, name='RY')
    return r

def rz(theta):
    r = np.array([[np.exp(1j * theta / 2), 0], [0, np.exp(-1j * theta / 2)]], dtype='complex128')
    r = tn.Node(r, name='RZ')
    return r

def u3(theta3):
    r = np.array([[np.cos(theta3[0] / 2) * np.exp(1j * theta3[1]) * np.exp(1j * theta3[2]),
                    -np.sin(theta3[0] / 2) * np.exp(1j * theta3[1]) * np.exp(-1j * theta3[2])],
                   [np.sin(theta3[0] / 2) * np.exp(-1j * theta3[1]) * np.exp(1j * theta3[2]),
                    np.cos(theta3[0] / 2) * np.exp(-1j * theta3[1]) * np.exp(-1j * theta3[2])]], dtype='complex128')

    r = tn.Node(r, name='U3')
    return r


def crx(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = rx(theta).tensor
    c = tn.Node(c, name='CRX')
    return c


def cry(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = ry(theta).tensor
    c = tn.Node(c, name='CRY')

    return c


def crz(theta):
    c = np.zeros((2, 2, 2, 2), dtype='complex128')
    c[0, 0] = np.eye(2)
    c[1, 1] = rz(theta).tensor
    c = tn.Node(c, name='CRX')

    return c


def cu3(theta3):
    c = np.zeros((2, 2, 2, 2))
    c[0, 0] = np.eye(2)
    c[1, 1] = u3(theta3).tensor
    c = tn.Node(c, name='CRX')

    return c


def rst(qubits):
    tens = np.array([1., 0.], dtype='complex128')
    for _ in range(qubits - 1):
        tens = np.tensordot(tens, np.array([1., 0.], dtype='complex128'), axes=0)

    tens = tn.Node(tens, name='Wavefunction')

    return tens

class QCircuit:
    def __init__(self, qubits):
        self.qubits = qubits
        self.psi = rst(self.qubits)
        self.psi_shape = self.psi.shape

    def reset(self):
        self.psi = rst(self.qubits)

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
        sum_list = list(range(1, 1 + self.qubits - len(measured_qubits), 1))
        con_list = sum_list.copy()
        index_list = list(range(-1, -1 - self.qubits, -1))
        for m_q, i in zip(measured_qubits, index_list):
            sum_list.insert(self.qubits - m_q - 1, i)

        return sum_list, con_list


    def H(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([h, self.psi], [l1, l2])

    def X(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([x, self.psi], [l1, l2])

    def Y(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([y, self.psi], [l1, l2])

    def Z(self, target):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([z, self.psi], [l1, l2])

    def RX(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([rx(theta), self.psi], [l1, l2])

    def RY(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([ry(theta), self.psi], [l1, l2])

    def RZ(self, target, theta):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([rz(theta), self.psi], [l1, l2])

    def U3(self, target, theta3):
        l1 = self.select_op_1(target)
        l2 = self.select_target_1(target)
        self.psi = tn.ncon([u3(theta3), self.psi], [l1, l2])

    def CX(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.psi = tn.ncon([cx, self.psi], [l1, l2])

    def CY(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.psi = tn.ncon([cy, self.psi], [l1, l2])

    def CZ(self, control, target):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)
        self.psi = tn.ncon([cz, self.psi], [l1, l2])

    def CRX(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.psi = tn.ncon([crx(theta), self.psi], [l1, l2])

    def CRY(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.psi = tn.ncon([cry(theta), self.psi], [l1, l2])

    def CRZ(self, control, target, theta):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.psi = tn.ncon([crz(theta), self.psi], [l1, l2])

    def CU3(self, control, target, theta3):
        l1 = self.select_op_2(control, target)
        l2 = self.select_target_2(control, target)

        self.psi = tn.ncon([cu3(theta3), self.psi], [l1, l2])


    def update_psi(self, tensor):
        """

        :param tensor: numpy array
        :return:
        """
        if tensor.shape != self.psi_shape:
            raise IndexError('Dimension mismatch')

        self.psi = tn.Node(tensor, name='Wavefunction')

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
        probabilities = np.abs(self.psi.tensor) ** 2
        return probabilities

    def measure_all(self, n_samples, return_probabilities = False):
        # Esto puede optimizarse utilizando tensornetworks??
        # probabilities = tn.Node(np.abs(self.psi.tensor) ** 2)
        probabilities = self.get_all_probabilities()
        measurements = self.measure_probs(probabilities, n_samples).reshape(probabilities.shape)

        if return_probabilities:
            return measurements, probabilities

        else:
            return measurements

    def get_some_probabilities(self, measured_qubits):
        probabilities = tn.Node(np.abs(self.psi.tensor) ** 2)
        # probabilities = np.abs(self.psi.tensor) ** 2
        contract_tensor = tn.Node(np.ones((2,) * (self.qubits - len(measured_qubits))))
        con_probabilities = tn.ncon([probabilities, contract_tensor], self.measure_list(measured_qubits)).tensor

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
        self.Domain_tensor = tn.Node(np.array([np.diag(np.array([x_i, 1, 1])) for x_i in domain]))
        self.F = f
        self.params_shape = (layers, 3)
        self.params = tn.Node(np.zeros(self.params_shape))
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
        self.params = tn.Node(new_parameters)
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
