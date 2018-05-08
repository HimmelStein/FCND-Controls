"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0


class NonlinearController(object):

    def __init__(self, kp_pq=0.01, kp_r=0.01, zkpd=31.0, kp_Yaw=1.0, kp_RollPitch=5.0,
                 xkp=0.0, xkd=0.0, ykp=0.0, ykd=0.0, k2d=0.00001):
        """Initialize the controller object and control gains"""
        self.z_k_p = zkpd #31.0 #90 #90=e0.99 88-eFail 105-e1.015 100-e1.036 120-eFail 80-eFail 60=e1.15 45-eFail 30-e1.30 15-eFail 1-eFail
        self.z_k_d = zkpd #31.0 #90 #90=e0.99 88-eFail 105-e1.015 100-e1.036 120-eFail 80-eFail 60=e1.15 45-eFail 30-e1.30 15-eFail 4-eFail

        # (p, d) fail when (20,10), (2,4), (3,4), (4,4), (1,6), (1,7), (1,9), (1,11), (1,12), (1,14), (1,16), (1,17)
        #                (2,4) (2,16), (2, 15), (2, 20) (10,20) (10, 90)
        # (1,4)-eV31, (1,5)-ev29, (1,8)-ev21, (1,10)-ev19, (1, 13)-ev31, (1,15)-ev24

        self.x_k_p = xkp # 0.20677777777777778 #xkp #0.20777777777777778 #0.2 #0.2111111111111111 0.24726315789473685 # 0.202
        self.x_k_d = xkd #0.4933222222222222 #0.4933222222222222 #kd #0.806 #kd
        self.y_k_p = ykp #0.3178888888888889 #kp #0.3188888888888889  #0.3188888888888889 #0.07868421052631579 #0.01902 #kp #0.02
        self.y_k_d = ykd #0.5902222222222222 #ykd #0.5892222222222222 #0.59 # 0.8394736842105263 #0.8347368421052631 #0.7878 #kd #
        self.x_k_2d = k2d #k2d

        # fail (20,20,5)
        # (5,5) 1-eV24, 2-eV33, 3-eV33, 4-fail, 5-fail
        # (8,8) 3-fail, 4-fail, 1.0-fail,
        # (10,10) 10-fail, 5-ev33, 1-eV27, 2-eV38, 3-fail
        # (15,15) 3.0-fail, 1.0-fail, 5-fail, 1.5-fail
        # (20,20) 1.0-fail, 5-fail, 10-fail
        # (30,30) 3-fail
        self.k_p_roll = kp_RollPitch #10.0
        self.k_p_pitch = kp_RollPitch # 10.0
        self.k_p_yaw = kp_Yaw #1.0

        # fail (90,90,30) (8,11,3) (23,23, 5) (110,110,15) (105,105,35)
        # (100, 100), 20-h25-v0.9, 30-h27, 40-h24, 50-fail
        self.k_p_p = kp_pq #100.0 # 80.0-h28-v1.02 105.0-h25-v0.98 100.0-h21-v0.94 50-eH35 55-h44v1.07
        self.k_p_q = kp_pq #100.0 # 100.0-h28-v1.02 105.0-h25-v0.98  100.0-h21-v0.94 50-eH35 55-h44v1.07
        self.k_p_r = kp_r  #35.0 # 35-h0.92-v27 30.0-h28-v1.02 40.0-h25-v0.98  30.0-h21-v0.94 50-eH35 15-h44v1.07

        self.g = GRAVITY

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        # ind_min points to the temporal point nearest to 'current_time' (can be in the future, or in the past)
        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        # ind_min points to the reference temporal point
        time_ref = time_trajectory[ind_min]

        # reference point is in the future
        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            # yaw_cmd triggers yaw_trajectory at ind_min-1, to move to the next index pint
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            # reference point is in the past or now
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1: # point to the last point, not next point
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                
                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        # if current_time == time1, position_cmd == position1
        # if current_time == time0, position_cmd == position0
        # position_cmd is the should-be position at the current_time
        position_cmd = (position1 - position0) * (current_time - time0) / (time1 - time0) + position0

        # the average velocity to move from position0 to position1 within the time of time1-time0
        velocity_cmd = (position1 - position0) / (time1 - time0)
        return (position_cmd, velocity_cmd, yaw_cmd)

    # first implement
    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2
            
        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        kp = np.array([self.k_p_p, self.k_p_q, self.k_p_r])
        rate = kp*(body_rate_cmd - body_rate)
        return rate
        # return np.array([0.0, 0.0, 0.0])

    # second implement
    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        # print('in roll-pitch-control')
        # print('thrust_cmd', thrust_cmd)
        thrust_cmd = np.clip(thrust_cmd, -MAX_THRUST,MAX_THRUST)
        print('thrust_cmd', thrust_cmd)
        roll, pitch, yaw = attitude
        rot_mat = euler2RM(roll, pitch, yaw)
        b_x_c, b_y_c = acceleration_cmd/thrust_cmd
        b_x_a, b_y_a = rot_mat[0][2], rot_mat[1][2]
        b_dot_x_c = self.k_p_roll * (b_x_c - b_x_a)
        b_dot_y_c = self.k_p_pitch * (b_y_c - b_y_a)
        p_c = np.divide(rot_mat[1][0] * b_dot_x_c - rot_mat[0][0] * b_dot_y_c, rot_mat[2][2])
        q_c = np.divide(rot_mat[1][1] * b_dot_x_c - rot_mat[0][1] * b_dot_y_c, rot_mat[2][2])
        # p_c = np.clip(p_c, 0, p_c)
        # q_c = np.clip(q_c, 0, q_c)
        return np.array([p_c, q_c])
        # return np.array([0.0,0.0])

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                                 acceleration_ff=np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        '''
        kpd = np.array([[self.x_k_p, self.x_k_d, 1],
                        [self.y_k_p, self.y_k_d, 1]])
        # print('kpd',kpd)
        mx = np.matrix([local_position_cmd - local_position,
                       local_velocity_cmd - local_velocity,
                       acceleration_ff]).transpose()
        # print('mx', mx)
        rlt = np.sum(np.multiply(kpd, mx), axis=1).transpose()
        rlt = np.asarray(rlt).reshape(-1)
        north, east= rlt

        '''
        x_target, y_target = local_position_cmd
        x_actual, y_actual = local_position
        x_dot_target, y_dot_target = local_velocity_cmd
        x_dot_actual, y_dot_actual = local_velocity
        print('acceleration_ff', acceleration_ff[0:2])
        x_dot_dot_target, y_dot_dot_target = acceleration_ff[0:2]
        print('local pos cmd', local_position_cmd)
        print('local pos', local_position)
        #print('local v cmd', local_velocity_cmd)
        #print('local v', local_velocity)
        #print("aff", acceleration_ff)
        #print('kp', self.x_k_p, self.y_k_p, 'kd', self.x_k_d, self.y_k_d)

        x_dot_dot_command = self.x_k_p * (x_target - x_actual) + self.x_k_d * (
                x_dot_target - x_dot_actual) + x_dot_dot_target*self.x_k_2d
        y_dot_dot_command = self.y_k_p * (y_target - y_actual) + self.y_k_d * (
                y_dot_target - y_dot_actual) + y_dot_dot_target*self.x_k_2d

        north = x_dot_dot_command
        east = y_dot_dot_command
        # print('raw north-east',np.array([north, east]))
        # if np.abs(north) > np.abs(east):
        #    s0 = np.abs(north)*10
        #else:
        #    s0 = np.abs(east)*10
        #north = np.clip(north, 0, 10*north/s0)
        #east = np.clip(east, 0, 10*east/s0)
        print(np.array([north, east]))

        return np.array([north, east])
        # return np.array([0.0, 0.0])

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude,
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """

        roll, pitch, yaw = attitude
        rot_mat = euler2RM(roll, pitch, yaw)
        u_bar_1 = self.z_k_p * (altitude_cmd - altitude) + self.z_k_d * (
                vertical_velocity_cmd - vertical_velocity) + acceleration_ff
        c = DRONE_MASS_KG*(u_bar_1 - GRAVITY) / rot_mat[2][2]
        return c
        # return np.array(0.0)

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        # print('in yaw-control')
        yawrate = self.k_p_yaw * (yaw_cmd - yaw)
        # yawrate = np.clip(yawrate, -0.01, 0.01)
        return yawrate



