from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag
from time import time
import pickle

def load_knn(filename):
    """
    Loads a stored KNN object saved with the string filename.
    Returns the loaded object.
    """
    if not isinstance(filename, str):
        raise ValueError("The filename must be a string.")
    if filename[-4:] != '.knn':
        filename = filename + '.knn'
    with open(filename, 'rb') as input:
        W, neuron, P = pickle.load(input)
    obj = KNN(W[0].shape[1]-1, W[1].shape[0], W[0].shape[0], neuron)
    obj.W, obj.P = W, P
    return obj

##########

class KNN:
    """
    Class for a feedforward neural network (NN). Currently only handles 1 hidden-layer,
    is always fully-connected, and uses the same activation function type for every neuron.
    The NN can be trained by extended kalman filter (EKF) or stochastic gradient descent (SGD).
    Use the train function to train the NN, the feedforward function to compute the NN output,
    and the classify function to round a feedforward to the nearest class values. A save function
    is also provided to store a KNN object in the working directory.
    """
    def __init__(self, nu, ny, nl, neuron, sprW=5):
        """
            nu: dimensionality of input; positive integer
            ny: dimensionality of output; positive integer
            nl: number of hidden-layer neurons; positive integer
        neuron: activation function type; 'logistic', 'tanh', or 'relu'
          sprW: spread of initial randomly sampled synapse weights; float scalar
        """
        # Function dimensionalities
        self.nu = int(nu)
        self.ny = int(ny)
        self.nl = int(nl)

        # Neuron type
        if neuron == 'logistic':
            self.sig = lambda V: (1 + np.exp(-V))**-1
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = lambda V: np.tanh(V)
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.float64(sigV > 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        self.neuron = neuron

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

####

    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.
        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)

####

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l.
        """
        U = np.float64(U)
        if U.ndim == 1 and len(U) > self.nu: U = U[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], U))
        h = self._affine_dot(self.W[1], l)
        if get_l: return h, l
        return h

####

    def classify(self, U, high, low=0):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        For each associated output, the closest integer between high
        and low is returned as a (m by ny) classification matrix.
        Basically, your training data should be (u, int_between_high_low).
        """
        return np.int64(np.clip(np.round(self.feedforward(U), 0), low, high))

####

    def train(self, nepochs, U, Y, method, P=None, Q=None, R=None, step=1, dtol=-1, dslew=1, pulse_T=-1):
        """
        nepochs: number of epochs (presentations of the training data); integer
              U: input training data; float array m samples by nu inputs
              Y: output training data; float array m samples by ny outputs
         method: extended kalman filter ('ekf') or stochastic gradient descent ('sgd')
              P: initial weight covariance for ekf; float scalar or (nW by nW) posdef array
              Q: process covariance for ekf; float scalar or (nW by nW) semiposdef array
              R: data covariance for ekf; float scalar or (ny by ny) posdef array
           step: step-size scaling; float scalar
           dtol: finish when RMS error avg change is <dtol (or nepochs exceeded); float scalar
          dslew: how many deltas over which to examine average RMS change; integer
        pulse_T: number of seconds between displaying current training status; float
        If method is 'sgd' then P, Q, and R are unused, so carefully choose step.
        If method is 'ekf' then step=1 is "optimal", R must be specified, and:
            P is None: P = self.P if self.P has been created by previous training
            Q is None: Q = 0
        If P, Q, or R are given as scalars, they will scale an identity matrix.
        Set pulse_T to -1 (default) to suppress training status display.
        Returns a list of the RMS errors at every epoch and a list of the covariance traces
        at every iteration. The covariance trace list will be empty if using sgd.
        """
        # Verify data
        U = np.float64(U)
        Y = np.float64(Y)
        if len(U) != len(Y):
            raise ValueError("Number of input data points must match number of output data points.")
        if (U.ndim == 1 and self.nu != 1) or (U.ndim != 1 and U.shape[-1] != self.nu):
            raise ValueError("Shape of U must be (m by nu).")
        if (Y.ndim == 1 and self.ny != 1) or (Y.ndim != 1 and Y.shape[-1] != self.ny):
            raise ValueError("Shape of Y must be (m by ny).")
        if Y.ndim == 1 and len(Y) > self.ny: Y = Y[:, np.newaxis]

        # Set-up
        if method == 'ekf':
            self.update = self._ekf

            if P is None:
                if self.P is None:
                    raise ValueError("Initial P not specified.")
            elif np.isscalar(P):
                self.P = P*np.eye(self.nW)
            else:
                if np.shape(P) != (self.nW, self.nW):
                    raise ValueError("P must be a float scalar or (nW by nW) array.")
                self.P = np.float64(P)

            if Q is None:
                self.Q = np.zeros((self.nW, self.nW))
            elif np.isscalar(Q):
                self.Q = Q*np.eye(self.nW)
            else:
                if np.shape(Q) != (self.nW, self.nW):
                    raise ValueError("Q must be a float scalar or (nW by nW) array.")
                self.Q = np.float64(Q)
            if np.any(self.Q): self.Q_nonzero = True
            else: self.Q_nonzero = False

            if R is None:
                raise ValueError("R must be specified for EKF training.")
            elif np.isscalar(R):
                self.R = R*np.eye(self.ny)
            else:
                if np.shape(R) != (self.ny, self.ny):
                    raise ValueError("R must be a float scalar or (ny by ny) array.")
                self.R = np.float64(R)
            if npl.matrix_rank(self.R) != len(self.R):
                raise ValueError("R must be positive definite.")

        elif method == 'sgd':
            self.update = self._sgd
        else:
            raise ValueError("The method argument must be either 'ekf' or 'sgd'.")
        last_pulse = 0
        RMS = []
        trcov = []

        # Shuffle data between epochs
        print("Training...")
        for epoch in xrange(nepochs):
            rand_idx = np.random.permutation(len(U))
            U_shuffled = U[rand_idx]
            Y_shuffled = Y[rand_idx]
            RMS.append(self.compute_rms(U, Y))

            # Check for convergence
            if len(RMS) > dslew and abs(RMS[-1] - RMS[-1-dslew])/dslew < dtol:
                print("\nConverged after {} epochs!\n\n".format(epoch+1))
                return RMS, trcov

            # Train
            for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

                # Forward propagation
                h, l = self.feedforward(u, get_l=True)

                # Do the learning
                self.update(u, y, h, l, step)
                if method == 'ekf': trcov.append(np.trace(self.P))

                # Heartbeat
                if (pulse_T >= 0 and time()-last_pulse > pulse_T) or (epoch == nepochs-1 and i == len(U)-1):
                    print("------------------")
                    print("  Epoch: {}%".format(int(100*(epoch+1)/nepochs)))
                    print("   Iter: {}%".format(int(100*(i+1)/len(U))))
                    print("   RMSE: {}".format(np.round(RMS[-1], 6)))
                    if method == 'ekf': print("tr(Cov): {}".format(np.round(trcov[-1], 6)))
                    print("------------------")
                    last_pulse = time()
        print("\nTraining complete!\n\n")
        RMS.append(self.compute_rms(U, Y))
        return RMS, trcov

####

    def _ekf(self, u, y, h, l, step):

        # Compute NN jacobian
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                       block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = step*K.dot(y-h)
        self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P = self.P - K.dot(H).dot(self.P)
        if self.Q_nonzero: self.P = self.P + self.Q

####

    def _sgd(self, u, y, h, l, step):
        e = h - y
        self.W[1] = self.W[1] - step*np.hstack((np.outer(e, l), e[:, np.newaxis]))
        D = (e.dot(self.W[1][:, :-1])*self.dsig(l)).flatten()
        self.W[0] = self.W[0] - step*np.hstack((np.outer(D, u), D[:, np.newaxis]))
        
if __name__ == '__main__':
    # Get some training data from the simulation of a nonlinear system, the Lorenz Attractor!
    dt = 0.01  # physical resolution
    tf = 100  # experiment duration
    T = np.arange(0, tf, dt, dtype=np.float64)  # time record
    X = np.zeros((len(T), 3), dtype=np.float64)  # state record
    Xdot = np.zeros_like(X)  # state derivative record
    x = np.array([1, 1, 1], dtype=np.float64)  # initial condition
    for i, t in enumerate(T):
        X[i] = np.copy(x)  # record
        Xdot[i] = np.array((10*(x[1]-x[0]),
                            x[0]*(28-x[2])-x[1],
                            x[0]*x[1]-2.6*x[2]))  # dynamic
        x = x + Xdot[i]*dt  # step simulation
    per = 0.01  # training data sampling period
    skip = int(per/dt)

    # Create and train KNN
    knn = KNN(nu=3, ny=3, nl=20, neuron='tanh')
    knn.train(nepochs=1, U=X[::skip], Y=Xdot[::skip], method='ekf', P=0.5, R=0.1, pulse_T=1)
  # knn.save("lorenz")
  # knn = load_knn('lorenz')

    # Use KNN to simulate system from same initial condition
    Xh = np.zeros_like(X)
    xh = X[0]
    for i, t in enumerate(T):
        Xh[i] = np.copy(xh)
        xh = xh + knn.feedforward(xh)*dt

  # Evaluation
    lim = int(1*len(T))
    fig1 = plt.figure()
    fig1.suptitle("Evolution", fontsize=22)
    ax = fig1.gca(projection='3d')
    ax.plot(X[0:lim:skip, 0], X[0:lim:skip, 1], X[0:lim:skip, 2], c='k', lw=1, ls=':', label="True")
    ax.plot(Xh[0:lim:skip, 0], Xh[0:lim:skip, 1], Xh[0:lim:skip, 2], c='m', lw=1, label="Predict")
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 30])
    ax.set_zlim([0, 50])
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_zlabel("z", fontsize=16)
    plt.legend()
    plt.show()
