# Piecewise Polynomial Spline interpolation for Trajectory Generation

Given a 3D occupancy map of an environment, efficiently compute feasible, minimum-snap trajectories that follow the shortest collision-free path from start to goal. The shortest, collision free path through the environment is found through the A* graph search algorithm. That path is pruned to a minimal set of waypoints and a sequence of polynomials is joined together to form a smooth trajectory.


## TODO:
- [ ] Create a FuncAnimation that smoothly transitions ax view_angle to view the paths easier rather than saving static image
- [ ] Create a branch where the trajectory is not piecewise but one continuous polynomial (Still debating if I actually want to do this)
- [ ] Create a quadratic program to ensure optimal smooth, collision-free path

## References
[1] Mellinger, Daniel, and Vijay Kumar. "Minimum snap trajectory generation and control for quadrotors." 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011.

[2] Richter, Charles, Adam Bry, and Nicholas Roy. "Polynomial trajectory planning for aggressive quadrotor flight in dense indoor environments." Robotics Research. Springer, Cham, 2016. 649-666.