import numpy as np

def filter_points(waypoints):
    waypoints = np.asarray(waypoints)
    # Initialize list of necessary waypoints
    necessary_waypoints = [waypoints[0]]

    # Check each triplet of waypoints for collinearity
    for i in range(1, waypoints.shape[0] - 1):
        prev_wp = necessary_waypoints[-1]
        curr_wp = waypoints[i]
        next_wp = waypoints[i + 1]

        # Calculate vectors
        vec1 = curr_wp - prev_wp
        vec2 = next_wp - curr_wp

        # Check if the vectors are collinear
        if not np.allclose(np.cross(vec1, vec2), 0):
            necessary_waypoints.append(curr_wp)

    necessary_waypoints.append(waypoints[-1])
    return np.array(necessary_waypoints)

class PolyTraj(object):
    def __init__(self, r, waypoints):
        '''
        Creates a piecewise polynomial spline to generate trajectories of an entity

                               | P1(t), t0 <= t <= t1
                        P(t) = | P2(t), t1 <= t <= t2
                               | P3(t), t2 <= t <= t3
                               
        Args:
            r (int): The order of the derivative to minimize
            waypoints (list, np.array): The waypoints the entity should pass through            
        '''
        self.r = r  ## order to minimize
        self.n = 2*r - 1  ## order of polynomial
        self.waypoints = filter_points(waypoints)
        self.num_waypoints = self.waypoints.shape[0]
        self.num_segments = self.num_waypoints - 1
        self.num_coefficients = self.n + 1

        ## Construct an array of time values where the values correspond to
        ## the time it should take the agent to reach that waypoint
        distances = np.zeros((self.waypoints.shape[0] - 1,3))
        for i in range(distances.shape[0]):
            distances[i] =  self.waypoints[i+1] - self.waypoints[i]
        self.times = np.cumsum(np.linalg.norm(distances, axis=1))
        self.times = np.append(np.array([[0]]), self.times)  ## Insert 0 at the beginning of self.times
        self.times /= 2

        ## Find the 3D rth-derivative of position
        self.d = {}  ## derivatives of the position
        for r in range(1, self.r):
            dr_array = np.zeros(self.waypoints.shape)
            for i in range(self.waypoints.shape[0] - 1):
                if (r == 1):
                    drdt = (self.waypoints[i+1] - self.waypoints[i]) / self.times[i+1]
                    dr_array[i+1] = drdt
                else:
                    drdt = (self.d[r-1][i + 1] - self.d[r-1][i]) / self.times[i+1]
                    dr_array[i+1] = drdt
            self.d[r] = dr_array

        for k in self.d.keys():
            self.d[k][-1] = np.zeros(3)  ## set end derivatives to 0

        self.coeffs = self.__solve_polynomials()  ## list of the coeffs for each segment

    def update(self, t):
        """
        Finds the position returned from the appropriate polynomial function created

        Args:
            t (float): Current time to find the position

        Return:
            np.array or None: Position at time 't' or No update
        """
        for i, time in enumerate(self.times):
            if t < time:
                segment = i - 1 if i > 0 else 0
                cx, cy, cz = self.coeffs[segment]
                x = self.__evaluate_trajectory(t - self.times[segment], cx)
                y = self.__evaluate_trajectory(t - self.times[segment], cy)
                z = self.__evaluate_trajectory(t - self.times[segment], cz)
                return np.array([x, y, z]).reshape((1,3))
        return None

    def __polynomial_basis(self, t, order=0):
        """
        Returns the polynomial function at time t or its derivative of a specified order.

        Args:
            t (float): The time value to substitute 't' in the function
            order (int, optional): The r-th derivative of the polynomial function

        Returns:
            list: A list of values after substituting in 't', before summation
        """
        basis = np.zeros(self.num_coefficients)
        temp = np.array([t**i for i in range(self.num_coefficients-order-1, -1, -1)])
        basis[:len(temp)] = temp
        return basis
    
    def __basis_coefficients(self, order=0):
        """
        Returns the coefficients of a polynomial function or its derivative of a specified order.
        
        Args:
            coefficients (list): A list of coefficients representing the polynomial function.
            order (int, optional): The order of the derivative to compute. Default is 0 (original polynomial).
        
        Returns:
            list: A list of coefficients for the polynomial function or its derivative of the specified order.
        """            
        coefficients = np.ones(self.n + 1)
        for _ in range(order):
            # Compute the derivative coefficients
            degree = len(coefficients) - 1
            derivative_coeffs = [coefficients[i] * i for i in range(1, degree + 1)]
            coefficients = derivative_coeffs
        
        return np.append(coefficients[::-1], np.zeros(self.num_coefficients - len(coefficients)))

    def __set_constraints(self, T, start, end):
        """
        Creates the A and b matrices that represents the start and end constraints of the polynomial function
        that contains up to a total of 2r constraints and has continuous constraints up to (r - 1)th-derivative order
        Note: This is done in 1D

        Args:
            T (float): The time it takes to travel from start to end
            start (list): The initial state at waypoint[i]
            end (list): The terminal state at waypoint[i+1]

        Return:
            A (np.array): Matrix of values to multiply against the vector of coefficients
            b (np.array): Vector of intial and terminal state values as constraints
        """
        start = np.array(start)
        end = np.array(end)
        A = np.zeros((self.num_coefficients, self.num_coefficients))
        b = np.zeros((self.num_coefficients, 1))
        b[::2] = start.reshape((-1, 1))
        b[1::2] = end.reshape((-1, 1))

        ## position constraints
        A[:2, :] = np.vander([0, T], N = self.n + 1)

        ## higher order derivative constraints
        order = 2
        row = 2
        for j in range((self.num_coefficients - 2)//2):
            order = j + 1
            A[row, -order-1] = np.polyder(np.poly1d([1]*self.num_coefficients), order)(0)
            A[row+1, :] = self.__basis_coefficients(order) * self.__polynomial_basis(T, order)
            row += 2

        return A, b
    
    def __solve_polynomials(self):
        """
        Finds the coefficients of each polynomial function found for every segment of the trajectory by 
        solving the Ap = b equation and finding A,b matrices from __set_constraints            
        """
        coeffs = []
        for i in range(self.num_segments):
            p = []
            for j in range(3):
                T = self.times[i+1]  ## time to reach next waypoint
                start = [self.waypoints[i, j]]
                end = [self.waypoints[i+1, j]]
                for k in self.d.keys():
                    start.append(self.d[k][i, j])
                    end.append(self.d[k][i + 1, j])
                A, b = self.__set_constraints(T - self.times[i], start, end)
                p.append(np.linalg.solve(A, b))
            coeffs.append(p)
        return coeffs

    def __evaluate_trajectory(self, t, coeffs):
        """
        Solves the polynomial in matrix form

        Args:
            t (float): The time to solve for
            coeffs (np.array): Coefficients for the polynomial

        Returns:
            float: The dot product between both vectors
        """
        return self.__polynomial_basis(t) @ coeffs
    

if __name__ == "__main__":
    waypoints = np.random.randint(0, 10, (5, 3))
    traj = PolyTraj(4, waypoints)
    time_span = np.linspace(0, traj.times[-1], 200)
    positions = []
    for t in time_span:
        pos = traj.update(t)
        if (np.any(pos) == None):
            pos = positions[-1]
        positions.append(pos)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    trajectory = np.asarray(positions).reshape((len(time_span), 3))

    # Plotting the trajectory and waypoints
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color='red', label='Waypoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("mytrajectory.png")
    plt.show(block=True)