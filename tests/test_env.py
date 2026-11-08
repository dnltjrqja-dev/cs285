import mujoco
import numpy as np
import pytest

from ppjuggle import ASSET_PATH, HOME_QPOS, EnvConfig, UR5PaddleJuggleEnv


@pytest.fixture
def env():
    return UR5PaddleJuggleEnv(seed=0)


def test_home_pose_paddle_faces_up(env):
    np.testing.assert_allclose(env.paddle_pos, [0.5, 0.0, 0.5], atol=1e-3)
    assert env.paddle_normal[2] > 0.9999


def test_arm_holds_home_pose(env):
    for _ in range(100):
        env.step(np.zeros(6))
    np.testing.assert_allclose(env.data.qpos[env.arm_qpos_adr], HOME_QPOS, atol=1e-3)


def _drop_on_paddle(height):
    """Drop the ball onto the static paddle; return (restitution, rebound height)."""
    env = UR5PaddleJuggleEnv(EnvConfig(drop_height=height, ball_xy_noise=0.0, ball_vel_noise=0.0))
    m, d = env.model, env.data
    face_z = env.paddle_pos[2]
    in_contact, v_in, zs = False, None, []
    for _ in range(int(1.5 / m.opt.timestep)):
        v = env.ball_vel[2]
        mujoco.mj_step(m, d)
        touching = env.ball_paddle_contact()
        if touching and not in_contact:
            v_in = v
        if in_contact and not touching:
            e = -env.ball_vel[2] / v_in
            break
        in_contact = touching
    for _ in range(int(0.6 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        zs.append(env.ball_pos[2] - face_z - 0.02)
    return e, max(zs)


def test_restitution_close_to_real_paddle():
    e, _ = _drop_on_paddle(0.305)
    assert 0.80 < e < 0.90


def test_rebound_height_with_drag():
    # e^2 * 0.305 = 0.22 without drag; air drag takes a few more cm off.
    _, h = _drop_on_paddle(0.305)
    assert 0.18 < h < 0.22


def test_terminal_velocity_matches_real_ball():
    m = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    d = mujoco.MjData(m)
    ball = m.joint("ball_free").qposadr[0]
    d.qpos[ball:ball + 3] = [3.0, 3.0, 100.0]
    for _ in range(5000):
        mujoco.mj_step(m, d)
    assert 8.0 < -d.qvel[m.joint("ball_free").dofadr[0] + 2] < 9.0


def test_rollout_matches_step_and_keeps_state():
    env = UR5PaddleJuggleEnv(EnvConfig(action_mode="absolute"), seed=1)
    s0 = env.get_state()
    H = 20
    rng = np.random.default_rng(0)
    ctrl = HOME_QPOS + 0.05 * rng.standard_normal((H, 6))

    states, sensors = env.rollout(ctrl[None])
    np.testing.assert_array_equal(env.get_state(), s0)  # rollout must not touch env.data

    for t in range(H):
        env.step(ctrl[t])
    np.testing.assert_allclose(states[0, -1], env.get_state(), atol=1e-8)
    sl = env.sensor_slices()
    np.testing.assert_allclose(sensors[0, -1, sl["ball_pos"]], env.ball_pos, atol=1e-8)


def test_episode_terminates_when_ball_drops(env):
    terminated = False
    for _ in range(2000):
        _, _, terminated, _, _ = env.step(np.zeros(6))
        if terminated:
            break
    assert terminated
