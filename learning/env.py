from gym_duckietown.simulator import Simulator


def launch_env():
    env = Simulator(
        seed=123,  # random seed
        map_name="zigzag_dists",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=True,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,  #4 start close to straight
        full_transparency=True,
        distortion=True,
        randomize_maps_on_reset=True,
    )

    return env
