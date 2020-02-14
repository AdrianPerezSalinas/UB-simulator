from __future__ import division
import numpy as np
import os
import math
import qupy
from qupy.operator import I, X

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))


if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np


def _to_tuple(x):
    if np.issubdtype(type(x), np.integer):
        x = (x,)
    return x


def _to_scalar(x):
    if xp != np:
        if isinstance(x, xp.ndarray):
            x = xp.asnumpy(x)
    if isinstance(x, np.ndarray):
        x = x.item(0)
    return x



class Qubits_error:
    """
    Creating qubits.
    Args:
        size (:class:`int`):
            Number of qubits.
        dtype:
            Data type of the data array.
        gpu (:class:`int`):
            GPU machine number.
    Attributes:
        data (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The state of qubits.
        size:
            Number of qubits.
        dtype:
            Data type of the data array.
    """

    def __init__(self, size, **kargs):

        self.size = size
        self.basic_operator = qupy.operator

        self.state = xp.zeros([2, 2] * self.size, dtype=dtype)
        self.state[tuple([0, 0] * self.size)] = 1

    def set_state(self, state):
        """set_state(self, state)
        Set state.
        Args:
            state (:class:`str` or :class:`list` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                If you set state as :class:`str`, you can set state :math:`|\\mathrm{state}\\rangle`
                (e.g. state='0110' -> :math:`|0110\\rangle`.)
                otherwise, qubit state is set that you entered as state.
        """
        if isinstance(state, str):
            if 2 * len(state) != self.state.ndim:
                raise ValueError('There were {} qubits prepared, but you specified {} qubits.'
                                 .format(self.state.ndim, len(state)))
            self.state = xp.zeros_like(self.state)
            l = [0] * 2 * self.size
            for i, _ in enumerate(state):
                l[i] = int(_)
                l[self.size + i] = int(_)
            self.state[tuple(l)] = 1
        else:
            self.state = xp.asarray(state, dtype=dtype)
            if self.state.ndim == 1 and len(state) == 2 ** self.size:
                self.state = xp.diag(self.state).reshape([2,2] * self.size)

    def apply_gate(self, state, operator, target):
        """gate(self, operator, target, control=None, control_0=None)
        Gate method.
        Args:
            operator (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Unitary operator
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operated qubits
            control (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 1
            control_0 (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 0
        """

        target = _to_tuple(target)
        '''
        operator = xp.asarray(operator, dtype=dtype)

        if operator.size != 2 ** (len(target) * 2):
            raise ValueError('You must set operator.size==2^(len(target)*2)')

        if operator.shape[0] != 2:
            operator = operator.reshape([2] * int(math.log2(operator.size)))

        print(control)
        if control is not None and control_0 is not None:
            print('holi')
            op = [I] * (len(control) + len(control_0)) + [operator] * len(target)
            print(op)

        elif control is not None and control_0 is None:
            
            c_idx = list(range(2 * self.size))
            t_idx = list(range(2 * self.size))
            for i, _c in enumerate(control):
                t_idx[_c] = 2 * self.size + 2 * i
                t_idx[self.size + _c] = 2 * self.size + 2 * i + 1

            i += 1
            for j, _t in enumerate(target):
                t_idx[_t] = 2 * self.size + 2 * i + 2 * j
                t_idx[self.size + _t] = 2 * self.size + 2 * i + 2 * j + 1

            o1_idx = list(range(2 * self.size, 2 * self.size + len(target) + len(control))) + list(control) + list(target)
            targ2 = [self.size + ct for ct in (control + target)]
            o2_idx = targ2 + list(range(2 * self.size + len(target) + len(control), 2 * self.size + 2 * (len(target) + len(control))))
            print(o1_idx)
            print(o2_idx)
            character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            o1_index = ''.join([character[i] for i in o1_idx])
            o2_index = ''.join([character[i] for i in o2_idx])
            c_index = ''.join([character[i] for i in c_idx])
            t_index = ''.join([character[i] for i in t_idx])
            subscripts = '{},{},{}->{}'.format(o1_index, c_index, o2_index, t_index)
            print(subscripts)
            print(op.shape)
            self.state = xp.einsum(subscripts, op, self.state, np.conjugate(op))

        elif control is None and control_0 is not None:
            print('holi')
            op = [I] * (len(control_0)) + [operator] * len(target)
            print(op)

        else:
        '''
        c_idx = list(range(2 * self.size))
        t_idx = list(range(self.size * 2))
        for i, _t in enumerate(target):
            t_idx[_t] = 2 * self.size + 2 * i
            t_idx[_t + self.size] = 2 * self.size + 2 * i + 1
        o1_idx = list(range(2 * self.size, 2 * self.size + len(target))) + list(target)
        targ2 = [self.size + t for t in target]
        o2_idx = list(targ2) + list(range(2 * self.size + len(target), 2 * self.size + 2 * len(target)))
        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o1_index = ''.join([character[i] for i in o1_idx])
        o2_index = ''.join([character[i] for i in o2_idx])
        c_index = ''.join([character[i] for i in c_idx])
        t_index = ''.join([character[i] for i in t_idx])
        subscripts = '{},{},{}->{}'.format(o1_index, c_index, o2_index, t_index)

        return xp.einsum(subscripts, operator, state, np.conjugate(operator))

    def perfect_gate(self, operator, target):
        self.state = self.apply_gate(self.state, operator, target)

    def noisy_gate(self, operators, probabilities, target):
        states = [[]] * len(operators)
        states[0] = self.perfect_gate(operators[0], target)
        for i, (o, p) in enumerate(zip(operators, probabilities)):
            states[i] = p * self.apply_gate(states[0], o, target)

        self.state = sum(states)

    def controlled_gate(self, operator, control, target):
        op = np.zeros((2, 2, 2, 2), dtype='complex128')
        op[0, 0] = np.eye(2)
        op[1, 1] = operator
        print(op)

        c_idx = list(range(2 * self.size))
        t_idx = list(range(self.size * 2))
        t_idx[target] = 2 * self.size
        t_idx[target + self.size] = 2 * self.size + 2
        t_idx[control] = 2 * self.size + 1
        t_idx[control + self.size] = 2 * self.size + 3
        o1_idx = [2 * self.size, control, 2*self.size + 1, target]
        o2_idx = [control + 2, 2 * self.size + 2, target + 2, 2 * self.size + 3]
        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o1_index = ''.join([character[i] for i in o1_idx])
        o2_index = ''.join([character[i] for i in o2_idx])
        c_index = ''.join([character[i] for i in c_idx])
        t_index = ''.join([character[i] for i in t_idx])
        subscripts = '{},{},{}->{}'.format(o1_index, c_index, o2_index, t_index)
        print(subscripts)

        self.state = xp.einsum(subscripts, op, self.state, np.conjugate(op))
