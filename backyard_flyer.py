# -*- coding: utf-8 -*-
"""
Solution to the Backyard Flyer Project.
"""

import time
import visdom
from enum import Enum
import os
import io
from contextlib import redirect_stdout
import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID

from unity_drone import UnityDrone
from controller import NonlinearController


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(UnityDrone):

    def __init__(self, connection, kp_pq=0.01, kp_r=0.01, zkpd=31, kp_Yaw=1.2, kp_RollPitch=4.0,
                 xkp=0.0, xkd=0.0, ykp=0.0, ykd=0.0, k2d=0.0, start=time.time(), maxTime=2000):
        super().__init__(connection)
        # Add a controller object
        self.controller = NonlinearController(kp_pq=kp_pq, kp_r=kp_r, zkpd=zkpd,
                                              kp_Yaw=kp_Yaw, kp_RollPitch=kp_RollPitch,
                                              xkp=xkp, xkd=xkd, ykp=ykp, ykd=ykd, k2d=k2d)
        self.start_time = start
        self.maxTime = maxTime

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.max_x_error = 0.0
        self.max_y_error = 0.0
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_POSITION, self.update_ne_plot_callback)
        # self.register_callback(MsgID.LOCAL_POSITION, self.update_d_plot_callback)

        self.register_callback(MsgID.STATE, self.state_callback)
        # == added ==
        self.register_callback(MsgID.ATTITUDE, self.attitude_callback)
        self.register_callback(MsgID.RAW_GYROSCOPE, self.gyro_callback)
        # self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)

        # == visdom
        self.v = visdom.Visdom()
        # Plot NE @https://udacity.github.io/udacidrone/docs/visdom-tutorial.html
        ne = np.array([0, 0]).reshape(-1, 2)
        self.ne_plot = self.v.scatter(ne,
                                      opts=dict(
                                          title="Delta (north, east)",
                                          xlabel='Delta in North',
                                          ylabel='Delta in East'
                                      ))

        # Plot D @https://udacity.github.io/udacidrone/docs/visdom-tutorial.html
        d0 = np.array([0])
        self.t = 1
        self.d_plot = self.v.line(d0, X=np.array([self.t]), opts=dict(
            title="Horizontal Errors (meters)",
            xlabel='Timestep',
            ylabel='Down'
        ))


    # ======== Added callback
    def attitude_callback(self):
        if self.flight_state == States.WAYPOINT:
            self.attitude_controller()

    def gyro_callback(self):
        if self.flight_state == States.WAYPOINT:
            self.bodyrate_controller()

    # ========= begin visdom callback
    def update_ne_plot_callback(self):
        """
        reference to https://udacity.github.io/udacidrone/docs/visdom-tutorial.html
        :return:
        """
        if self.flight_state == States.TAKEOFF:
            ne = np.array([self.target_position[0] - self.local_position[0],
                           self.target_position[1] - self.local_position[1]]).reshape(-1, 2)
            self.v.scatter(ne, win=self.ne_plot, update='append')

    def update_d_plot_callback(self):
        """
        reference to https://udacity.github.io/udacidrone/docs/visdom-tutorial.html
        :return:
        """
        target0, target1 = self.target_position[0:2]
        loc0, loc1 = self.local_position[0:2]
        d0 = np.array([np.sqrt((-loc0+target0)**2 + (-loc1+target1)**2)])

        # d = np.array([np.linalg.norm(target - np.array(self.local_position[0:2]))])
        print('delta x y', loc0-target0, loc1-target1)

        print(d0)
        # update timestep
        self.t += 1
        self.v.line(d0, X=np.array([self.t]), win=self.d_plot, update='append')


    # ======== Begin of controller interface ========

    def position_controller(self):
        """Sets the local acceleration target using the local position and local velocity"""
        # print('in position controllter')
        (self.local_position_target, self.local_velocity_target, yaw_cmd) = self.controller.trajectory_control(
            self.position_trajectory, self.yaw_trajectory, self.time_trajectory, time.time())
        self.attitude_target = np.array((0.0, 0.0, yaw_cmd))

        affx, affy = 0.0, 0.0
        if self.waypoint_number -1 >=0:
            x1, y1 = -self.position_trajectory[self.waypoint_number -1][0:2]
            x2, y2 = -self.local_position[0:2]
            x3, y3 = -self.local_position_target[0:2]
            aff = 2*y1/(x2-x1)/(x3-x1)-2*y2/(x3-x2)/(x2-x1) + 2*y3/(x3-x2)/(x3/x1)
            l = np.linalg.norm(np.array([x1,y1])-np.array([x3,y3]))
            affx = aff*(x3-x1)/l
            affy = aff*(y3-y1)/l

        acceleration_cmd = self.controller.lateral_position_control(-self.local_position_target[0:2],
                                                                    -self.local_velocity_target[0:2],
                                                                    -self.local_position[0:2],
                                                                    -self.local_velocity[0:2],
                                                                    acceleration_ff=[affx,affy])

        target0, target1 = self.target_position[0:2]
        loc0, loc1 = self.local_position[0:2]
        if self.max_x_error < np.abs(loc0 - target0):
            self.max_x_error = np.abs(loc0 - target0)
        if self.max_y_error < np.abs(loc1 - target1):
            self.max_y_error = np.abs(loc1 - target1)
        print('max x error', self.max_x_error)
        print('max y error', self.max_y_error)

        self.local_acceleration_target = np.array([acceleration_cmd[0], acceleration_cmd[1], 0.0])

    def attitude_controller(self):
        """Sets the body rate target using the acceleration target and attitude"""
        self.thrust_cmd = self.controller.altitude_control(-self.local_position_target[2],
                                                           -self.local_velocity_target[2],
                                                           -self.local_position[2],
                                                           -self.local_velocity[2],
                                                           self.attitude, 9.81)
        roll_pitch_rate_cmd = self.controller.roll_pitch_controller(self.local_acceleration_target[0:2], self.attitude,
                                                                      self.thrust_cmd)
        yawrate_cmd = self.controller.yaw_control(self.attitude_target[2], self.attitude[2])
        self.body_rate_target = np.array([roll_pitch_rate_cmd[0], roll_pitch_rate_cmd[1], yawrate_cmd])

    def bodyrate_controller(self):
        """Commands a moment to the vehicle using the body rate target and body rates"""

        # self.thrust_cmd = self.controller.altitude_control(-self.local_position_target[2],
        #                                                   -self.local_velocity_target[2],
        #                                                   -self.local_position[2],
        #                                                   -self.local_velocity[2],
        #                                                   self.attitude, 9.81)


        moment_cmd = self.controller.body_rate_control(self.body_rate_target, self.gyro_raw)
        self.cmd_moment(moment_cmd[0], moment_cmd[1], moment_cmd[2], self.thrust_cmd)

    # ======== End ========

    def local_position_callback(self):
        current = time.time()
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                # replace
                # self.all_waypoints = self.calculate_box()
                # with
                (self.position_trajectory, self.time_trajectory, self.yaw_trajectory) = self.load_test_trajectory(
                    time_mult=1.5)
                self.all_waypoints = self.position_trajectory.copy()
                self.waypoint_number = -1

                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            # replace
            # if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
            # with
            if time.time() > self.time_trajectory[self.waypoint_number]:
                if len(self.all_waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0 or current - self.start_time > self.maxTime:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.WAYPOINT:
            # pass
            self.position_controller()

        if self.flight_state == States.LANDING:
            current = time.time()
            # print(self.start_time, current, self.maxTime)
            if self.global_position[2] - self.global_home[2] < 0.1 or current - self.start_time > self.maxTime:
                if abs(self.local_position[2]) < 0.01 or current - self.start_time > self.maxTime:
                    self.disarming_transition()
            # added
            if self.global_position[2] < -5.0:
                self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def calculate_box(self):
        print("Setting Home")
        local_waypoints = [[10.0, 0.0, 3.0], [10.0, 10.0, 3.0], [0.0, 10.0, 3.0], [0.0, 0.0, 3.0]]
        return local_waypoints

    def arming_transition(self):
        print("arming transition")
        self.take_control()
        self.arm()
        self.set_home_position(self.global_position[0], self.global_position[1],
                               self.global_position[2])  # set the current location to be the home position

        self.flight_state = States.ARMING

    def takeoff_transition(self):
        print("takeoff transition")
        # self.global_home = np.copy(self.global_position)  # can't write to this variable!
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        # print("waypoint transition")
        self.target_position = self.all_waypoints.pop(0)
        # print('target position', self.target_position)
        # disable self.cmd_position
        # self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], 0.0)
        # replace with self.local_position_target
        self.local_position_target = np.array((self.target_position[0],
                                               self.target_position[1],
                                               self.target_position[2]))
        # for testing
        # self.local_position_target = np.array([0.0, 0.0, -3.0])
        self.flight_state = States.WAYPOINT
        # add
        self.waypoint_number = self.waypoint_number + 1

    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print("manual transition")
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        # self.connect()

        print("starting connection")
        # self.connection.start()

        super().start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


def gear_bodyrate():
    kp_r = 0.01
    while kp_r < 30.0:
        window = 4
        mid = kp_r * 10 - window
        kp_pq = kp_r
        # kp_pq = float(random.choice(range(int(mid-window), int(mid +window), 1)))
        while kp_pq <= mid + window:
            print('trying kp_pq,kp_r: ', kp_pq, kp_r)
            conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
            # conn = WebSocketConnection('ws://127.0.0.1:5760')
            drone = BackyardFlyer(conn, kp_pq=kp_pq, kp_r=kp_r, start=time.time(), maxTime=25)
            time.sleep(2)
            drone.start()
            f = io.StringIO()
            with redirect_stdout(f):
                drone.print_mission_score()
            out = f.getvalue()
            print(out)
            with open('success_log0.txt', 'a+') as ifh:
                ifh.write(out)
                ifh.write('kp_pq: {} kp_r: {}\n'.format(kp_pq, kp_r))
            cmd = """
                osascript -e 'tell application "FCND-Control_MacOS" 
                    activate
                    tell application "System Events"  to keystroke "R" using {shift down} 
                end tell' 
                """
            os.system(cmd)
            time.sleep(2)
            os.system(cmd)
            kp_pq += 0.010
            time.sleep(2)
        kp_r += 1.0


def gear_altitude():
    z_k_pr = 1
    while z_k_pr < 100.0:
        print('trying z_k_pr : ',z_k_pr)
        conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
        # conn = WebSocketConnection('ws://127.0.0.1:5760')
        drone = BackyardFlyer(conn, zkpd=z_k_pr, start=time.time(), maxTime=25)
        time.sleep(2)
        drone.start()
        f = io.StringIO()
        with redirect_stdout(f):
            drone.print_mission_score()
        out = f.getvalue()
        print(out)
        with open('success_log0.txt', 'a+') as ifh:
            ifh.write(out)
            ifh.write('z_k_pr: {} \n'.format(z_k_pr))
        cmd = """
            osascript -e 'tell application "FCND-Control_MacOS" 
                activate
                tell application "System Events"  to keystroke "R" using {shift down} 
            end tell' 
            """
        os.system(cmd)
        time.sleep(2)
        os.system(cmd)
        z_k_pr += 10
        time.sleep(2)


def gear_yaw_roll_pitch():
    delta, kp_yaw, kp_roll_pitch = 1.0, 1.0, 10.0
    while kp_roll_pitch < 30.0:
        while kp_yaw <= kp_roll_pitch:
            print('trying kp_yaw, kp_roll_pitch: ', kp_yaw, kp_roll_pitch)
            conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
            # conn = WebSocketConnection('ws://127.0.0.1:5760')
            drone = BackyardFlyer(conn, kp_Yaw=kp_yaw, kp_RollPitch=kp_roll_pitch, start=time.time(), maxTime=25)
            time.sleep(2)
            drone.start()
            f = io.StringIO()
            with redirect_stdout(f):
                drone.print_mission_score()
            out = f.getvalue()
            print(out)
            with open('success_log0.txt', 'a+') as ifh:
                ifh.write(out)
                ifh.write('kp_yaw: {} kp_roll_pitch: {}\n'.format(kp_yaw, kp_roll_pitch))
            cmd = """
                osascript -e 'tell application "FCND-Control_MacOS" 
                    activate
                    tell application "System Events"  to keystroke "R" using {shift down} 
                end tell' 
                """
            os.system(cmd)
            time.sleep(2)
            os.system(cmd)
            kp_yaw += delta
            time.sleep(2)
        kp_roll_pitch += 10.0
        kp_yaw = kp_roll_pitch / 10.0
        delta = kp_yaw


def gear_k2d_control():
    for k2d in list(np.linspace(0.0, 1.0, num=10)):
        print('trying k2d: ', k2d)
        conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
        # conn = WebSocketConnection('ws://127.0.0.1:5760')
        drone = BackyardFlyer(conn, k2d=k2d, start=time.time(), maxTime=50)
        time.sleep(2)
        drone.start()
        f = io.StringIO()
        with redirect_stdout(f):
            drone.print_mission_score()
        out = f.getvalue()
        print(out)
        with open('success_log0.txt', 'a+') as ifh:
            ifh.write(out)
            ifh.write(' k2d: {} max x error {} max y error {}\n'.format(k2d, drone.max_x_error, drone.max_y_error))
        cmd = """
            osascript -e 'tell application "FCND-Control_MacOS" 
                activate
                tell application "System Events"  to keystroke "R" using {shift down} 
            end tell' 
            """
        os.system(cmd)
        time.sleep(2)
        os.system(cmd)
        time.sleep(2)


def gear_xypd_parameters(xkp0=0.0, xkp1=0.0, xkpnum=0,
                         xkd0=0.0, xkd1=0.0, xkdnum=0,
                         ykp0=0.0, ykp1=0.0, ykpnum=0,
                         ykd0=0.0, ykd1=0.0, ykdnum=0):
    for xkp in list(np.linspace(xkp0, xkp1, num=xkpnum)):
        for xkd in list(np.linspace(xkd0, xkd1, num=xkdnum)):
            for ykp in list(np.linspace(ykp0, ykp1, num=ykpnum)):
                for ykd in list(np.linspace(ykd0, ykd1, num=ykdnum)):
                    print('trying xkp {}, xkd {}, ykp {}, ykd {}'.format(xkp, xkd, ykp, ykd))
                    conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
                    # conn = WebSocketConnection('ws://127.0.0.1:5760')
                    drone = BackyardFlyer(conn, xkp=xkp, xkd=xkd, ykp=ykp, ykd=ykd, start=time.time(), maxTime=50)
                    time.sleep(2)
                    drone.start()
                    f = io.StringIO()
                    with redirect_stdout(f):
                        drone.print_mission_score()
                    out = f.getvalue()
                    print(out)
                    with open('success_log0.txt', 'a+') as ifh:
                        ifh.write(out)
                        ifh.write('xkp {}, xkd {}, ykp {}, ykd {} max x error {} max y error {}\n'.format(xkp, xkd, ykp,
                                                                                                          ykd,
                                                                                                drone.max_x_error,
                                                                                                drone.max_y_error))
                    cmd = """
                            osascript -e 'tell application "FCND-Control_MacOS" 
                            activate
                            tell application "System Events"  to keystroke "R" using {shift down} 
                            end tell' 
                        """
                    if 'True' in cmd:
                        return
                    os.system(cmd)
                    time.sleep(2)
                    os.system(cmd)
                    time.sleep(2)


if __name__ == "__main__":
    conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://127.0.0.1:5760')

    # gear_bodyrate()
    # gear_altitude()
    # gear_yaw_roll_pitch()

    need_gear = False
    if need_gear:
        gear_xypd_parameters(xkp0=0.19577777777777777-0.01, xkp1=0.19577777777777777+0.01, xkpnum=10, #0.205
                         xkd0=0.4723222222222222-0.01, xkd1=0.4723222222222222+0.01, xkdnum=10, #0.492
                         ykp0=0.31094736842105264-0.01, ykp1=0.31094736842105264+0.01, ykpnum=10, #0.31
                         ykd0=0.5907777777777777-0.01, ykd1=0.5907777777777777+0.01, ykdnum=10, #0.583
                        )

    # gear_k2d_control()

    drone = BackyardFlyer(conn,
                          xkp=0.18577777777777776, xkd=0.4623222222222222,
                          ykp=0.30094736842105263, ykd=0.5852222222222222,
                          start=time.time(), maxTime=30)
    time.sleep(2)
    drone.start()
    # added
    drone.print_mission_score()

    cmd = """
    osascript -e 'tell application "FCND-Control_MacOS" 
            activate
            tell application "System Events"  to keystroke "R" using {shift down} 
        end tell' 
    """
    os.system(cmd)
