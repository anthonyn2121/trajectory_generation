import numpy as np

class PolyTraj(object):
    def __init__(self, n, waypoints):
        self.n = n  ## order of polynomial
        self.waypoints = waypoints
        print("waypoints: \n", waypoints)
        self.num_waypoints = waypoints.shape[0]
        self.num_segments = self.num_waypoints - 1
        self.num_coefficients = self.n  + 1


        ## Construct an array of time values where the values correspond to
        ## the time it should take the agent to reach that waypoint
        distances = np.zeros((waypoints.shape[0] - 1,3))
        for i in range(distances.shape[0]):
            distances[i] =  waypoints[i+1] - waypoints[i]
        self.times = np.cumsum(np.linalg.norm(distances, axis=1))
        self.times = np.append(np.array([[0]]), self.times)  ## Insert 0 at the beginning of self.times
        self.times /= 2
        print("times: ", self.times)

        self.coeffs = self.__solve_polynomials()  ## list of the coeffs for each segment
        print(len(self.coeffs))
        print(len(self.coeffs[0]))
        # ## Construct a Gaussian curve that will be the velocity profile of the agent
        # vmax = 5  ## m/s
        # tpeak = (self.times[-1] - 0) / 2  ## velocity should peak at middle of trajectory
        # sigma = (self.times[-1] - 0) / 5
        # self.v = vmax * np.exp(-((self.times - tpeak) ** 2) / (2 * sigma ** 2))
        # self.v[0] = 0
        # self.v[-1] = 0
        # print("velocities: ", self.v)

        # ## Construct array of accelerations
        # self.a = np.gradient(self.v, self.times)
        # self.a[0] = 0
        # self.a[-1] = 0
        # print("accelerations: ", self.a)

        # coefficients = []
        # for axis in range(3):
        #     for i in range(self.num_segments - 1):
        #         dt = self.times[i+1] - self.times[i]
        #         start = self.waypoints[i, axis]
        #         end = self.waypoints[i+1, axis]
        #         A, b = self.__set_constraints(dt, start, end)
        #         p = np.linalg.solve(A, b)
        #         coefficients.append(p)
        # print(len(p))

    def __polynomial_basis(self, t, order=0):
        basis = np.zeros(self.num_coefficients)
        temp = np.array([t**i for i in range(self.num_coefficients-order-1, 0, -1)])
        basis[:len(temp)] = temp
        return basis
    
    def __basis_coefficients(self, order):
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
        A = np.zeros((self.num_coefficients, self.num_coefficients))
        b = np.zeros((self.num_coefficients, 1))

        ## position constraints
        V = np.vander([0, T], N = self.n + 1)
        A[:2, :] = np.vander([0, T], N = self.n + 1)
        b[:2] = np.array([start, end]).reshape((2,1))

        order = 2
        row = 2
        # for j in range(2):
        for j in range((self.num_coefficients - 2)//2):
            order = j + 1
            # print("row: ",row, "order: ",order)
            A[row, -order-1] = np.polyder(np.poly1d([1]*self.num_coefficients), order)(0)
            # print("basis_coeffs: ", self.__basis_coefficients(order), " basis: ", [i for i in range(self.num_coefficients-order-1, -1, -1)])
            A[row+1, :] = self.__basis_coefficients(order) * self.__polynomial_basis(T, order)
            row += 2

        # print("A: \n", A)
        # print("b: \n", b)

        return A, b
    
    def __solve_polynomials(self):
        x, y, z = self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]

        coeffs = []
        for i in range(self.num_segments):
            p = []
            for axis in [x, y, z]:
                T = self.times[i+1]  ## time to reach next waypoint
                A, b = self.__set_constraints(T, axis[i], axis[i+1])
                p.append(np.linalg.solve(A, b))
            coeffs.append(p)
        return coeffs

    def update(self, t):
        for i, time in enumerate(self.times):
            # print("t: ", t, " time waypoint: ", time)
            if t < time:
                # print("Finding position")
                segment = i - 1
                cx, cy, cz = self.coeffs[segment]
                x = self.__evaluate_trajectory(t, cx)
                y = self.__evaluate_trajectory(t, cy)
                z = self.__evaluate_trajectory(t, cz)
                return (x, y, z)


    def __evaluate_trajectory(self, t, coeffs):
        return self.__polynomial_basis(t) @ coeffs
    

if __name__ == "__main__":
    traj = PolyTraj(5, np.random.randint(0, 10, ((5,3))))
    time_span = np.linspace(0, 15, 200)
    for t in time_span:
        print("Updated position: ", traj.update(t))