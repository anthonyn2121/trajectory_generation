import numpy as np

class PolyTraj(object):
    def __init__(self, r, waypoints):
        self.r = r  ## 
        self.n = 2*r - 1  ## order of polynomial
        self.waypoints = np.asarray(waypoints)
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
        # self.times /= 2
        print("times: ", self.times)

        ## Find velocity by finding distance/time
        # self.v = np.zeros(self.waypoints.shape)
        # for i in range(self.waypoints.shape[0] - 1):
        #     v = (self.waypoints[i+1] - self.waypoints[i]) / self.times[i+1] 
        #     self.v[i+1] = v
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

        print(self.d)

        # self.v[-1] = np.zeros(3)
        # print("velocities: ", self.v)
        # print(self.v[3,2])

        # A, b = self.__set_constraints(self.times[1], self.waypoints[1][1], self.waypoints[2][1])
        # print("A: \n", A)
        # print("b: \n", b)
        self.coeffs = self.__solve_polynomials()  ## list of the coeffs for each segment

    def __polynomial_basis(self, t, order=0):
        basis = np.zeros(self.num_coefficients)
        temp = np.array([t**i for i in range(self.num_coefficients-order-1, -1, -1)])
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
        # x_t, vx_t = start
        # x_T, vx_T = end
        start = np.array(start)
        end = np.array(end)
        A = np.zeros((self.num_coefficients, self.num_coefficients))
        b = np.zeros((self.num_coefficients, 1))
        b[::2] = start.reshape((-1, 1))
        b[1::2] = end.reshape((-1, 1))

        ## position constraints
        A[:2, :] = np.vander([0, T], N = self.n + 1)
        # b[:4] = np.array([x_t, x_T, vx_t, vx_T]).reshape((4,1))

        order = 2
        row = 2
        for j in range((self.num_coefficients - 2)//2):
            order = j + 1
            A[row, -order-1] = np.polyder(np.poly1d([1]*self.num_coefficients), order)(0)
            A[row+1, :] = self.__basis_coefficients(order) * self.__polynomial_basis(T, order)
            row += 2

        # print("A: \n", A)
        # print("b: \n", b)

        return A, b
    
    def __solve_polynomials(self):
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

    def update(self, t):
        for i, time in enumerate(self.times):
            if t < time:
                segment = i - 1 if i > 0 else 0
                cx, cy, cz = self.coeffs[segment]
                x = self.__evaluate_trajectory(t - self.times[segment], cx)
                y = self.__evaluate_trajectory(t - self.times[segment], cy)
                z = self.__evaluate_trajectory(t - self.times[segment], cz)
                return np.array([x, y, z]).reshape((1,3))
        return None

    def __evaluate_trajectory(self, t, coeffs):
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