import cv2
from env import launch_env
from _loggers import Logger
from utils.helpers import SteeringToWheelVelWrapper

EPISODES = 25
STEPS = 2000

DEBUG = True
env = launch_env()
wrapper = SteeringToWheelVelWrapper()
logger = Logger(env, log_file='train-v10.log')

for episode in range(0, EPISODES):
    prev_angles = [0] * 10
    prev_angle = 0
    for steps in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        # k_p = 17
        # k_d = 9
        # k_i = 0.1
        # if -0.5 < lane_pose.angle_deg < 0.5:
        #     speed = 1
        # elif -1 < lane_pose.angle_deg < 1:
        #     speed = 0.9
        # elif -2 < lane_pose.angle_deg < 2:
        #     speed = 0.8
        # elif -20 < lane_pose.angle_deg < 20:
        #     speed = 0.6
        # elif -30 < lane_pose.angle_deg < 30:
        #     speed = 0.4
        # else:
        #     k_p = 33
        #     k_d = 8
        #     k_i = 0.05
        #     speed = 0.3

        k_p = 17
        k_d = 9
        k_i = 0.1
        if -0.5 < lane_pose.angle_deg < 0.5:
            speed = 1
        elif -1 < lane_pose.angle_deg < 1:
            speed = 0.9
        elif -2 < lane_pose.angle_deg < 2:
            speed = 0.8
        elif -10 < lane_pose.angle_deg < 10:
            speed = 0.5
        else:
            speed = 0.3

        prev_angles.append(abs(prev_angle - lane_pose.angle_deg))
        prev_angles.pop(0)
        prev_angle = lane_pose.angle_deg

        # angle of the steering wheel, which corresponds to the angular velocity in rad/s
        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads + k_i * sum(prev_angles)

        # Convert to wheel velocities
        action = wrapper.convert([speed, steering])
        observation, reward, done, info = env.step(action)
        closest_point, _ = env.closest_curve_point(env.cur_pos, env.cur_angle)
        if closest_point is None:
            done = True
            break
        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

        logger.log(observation, action, reward, done, info)
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

logger.close()
env.close()
