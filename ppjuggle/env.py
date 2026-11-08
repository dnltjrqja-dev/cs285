"""MuJoCo environment: UR5 arm with a table-tennis paddle as end effector, plus a ball.

The environment is deliberately controller-agnostic so it can back an MPC
controller (sampling-based or gradient-based) as well as RL:

* ``step`` / ``reset`` follow the Gymnasium convention but need no Gymnasium.
* ``get_state`` / ``set_state`` snapshot the full physics state, so a planner
  can branch from the current state.
* ``rollout`` simulates many open-loop control sequences in parallel with
  ``mujoco.rollout`` (multi-threaded C++), which is what an MPPI/CEM planner needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSET_PATH = Path(__file__).parent / "assets" / "ur5_paddle.xml"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Joint angles that put the paddle center at (0.5, 0.0, 0.5) m with its face
# pointing straight up (+z). Found with damped least-squares IK.
HOME_QPOS = np.array([-0.2201, -1.0809, 1.3031, 1.3486, -1.5708, 0.0])

BALL_RADIUS = 0.02
FULL_STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


@dataclass
class EnvConfig:
    frame_skip: int = 10              # physics steps per control step (1 ms each -> 100 Hz control)
    episode_seconds: float = 20.0
    action_mode: str = "delta"        # "delta": action = change of joint targets; "absolute": joint targets
    max_delta: float = 0.05           # rad per control step when action_mode == "delta"
    drop_height: float = 0.30         # initial ball height above the paddle face [m]
    ball_xy_noise: float = 0.01       # std of initial ball xy offset from the paddle center [m]
    ball_vel_noise: float = 0.05      # std of initial ball velocity [m/s]
    target_apex: float = 0.30         # desired apex height above the paddle face [m] (used by reward)
    fail_below: float = 0.15          # episode ends if ball drops this far below the paddle face [m]


class UR5PaddleJuggleEnv:
    """UR5 + paddle + ping-pong ball.

    Observation (27,):
        arm qpos (6), arm qvel (6),
        ball pos (3), ball vel (3),
        paddle center pos (3), paddle normal (3), paddle linear vel (3)

    Action (6,): joint position targets (absolute or delta, see ``EnvConfig``)
        tracked by the PD position actuators in the MJCF.
    """

    obs_dim = 27
    act_dim = 6

    def __init__(self, config: EnvConfig | None = None, xml_path: str | Path = ASSET_PATH, seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        m = self.model
        self.arm_qpos_adr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof_adr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        ball_jnt = m.joint("ball_free")
        self.ball_qpos_adr = ball_jnt.qposadr[0]
        self.ball_dof_adr = ball_jnt.dofadr[0]
        self.paddle_site = m.site("paddle_center").id
        self.paddle_geom = m.geom("paddle_blade").id
        self.ball_geom = m.geom("ball_geom").id
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()

        self.dt = m.opt.timestep * self.cfg.frame_skip
        self.max_steps = int(round(self.cfg.episode_seconds / self.dt))
        self.nstate = mujoco.mj_stateSize(m, FULL_STATE)
        self._rollout_data: list[mujoco.MjData] = []
        self._renderer: mujoco.Renderer | None = None
        self.reset()

    # ------------------------------------------------------------------ core API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d, c = self.model, self.data, self.cfg
        mujoco.mj_resetData(m, d)

        d.qpos[self.arm_qpos_adr] = HOME_QPOS
        d.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(m, d)

        paddle_pos = d.site_xpos[self.paddle_site].copy()
        ball_pos = paddle_pos + np.array([0.0, 0.0, BALL_RADIUS + c.drop_height])
        ball_pos[:2] += self.rng.normal(0.0, c.ball_xy_noise, size=2)
        b = self.ball_qpos_adr
        d.qpos[b:b + 3] = ball_pos
        d.qpos[b + 3:b + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self.ball_dof_adr:self.ball_dof_adr + 3] = self.rng.normal(0.0, c.ball_vel_noise, size=3)
        mujoco.mj_forward(m, d)

        self.steps = 0
        self.num_hits = 0
        self._in_contact = False
        return self._get_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if self.cfg.action_mode == "delta":
            target = self.data.ctrl + np.clip(action, -1.0, 1.0) * self.cfg.max_delta
        else:
            target = action
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

        hit = False
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            touching = self.ball_paddle_contact()
            if touching and not self._in_contact:
                hit = True
                self.num_hits += 1
            self._in_contact = touching
        self.steps += 1

        obs = self._get_obs()
        terminated = self.ball_dropped()
        truncated = self.steps >= self.max_steps
        reward = self.reward(obs, hit, terminated)
        info = self._info()
        info["hit"] = hit
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _get_obs(self):
        d = self.data
        b, bv = self.ball_qpos_adr, self.ball_dof_adr
        paddle_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, self.paddle_site, paddle_vel, 0)
        return np.concatenate([
            d.qpos[self.arm_qpos_adr],
            d.qvel[self.arm_dof_adr],
            d.qpos[b:b + 3],
            d.qvel[bv:bv + 3],
            d.site_xpos[self.paddle_site],
            d.site_xmat[self.paddle_site].reshape(3, 3)[:, 2],
            paddle_vel[3:],  # [angular; linear] -> linear part, world frame
        ])

    def _info(self):
        return {"num_hits": self.num_hits, "time": self.data.time}

    @property
    def ball_pos(self):
        return self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3].copy()

    @property
    def ball_vel(self):
        return self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3].copy()

    @property
    def paddle_pos(self):
        return self.data.site_xpos[self.paddle_site].copy()

    @property
    def paddle_normal(self):
        return self.data.site_xmat[self.paddle_site].reshape(3, 3)[:, 2].copy()

    def ball_paddle_contact(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if {g1, g2} == {self.ball_geom, self.paddle_geom}:
                return True
        return False

    def ball_dropped(self) -> bool:
        return self.ball_pos[2] < self.paddle_pos[2] - self.cfg.fail_below

    def reward(self, obs, hit: bool, terminated: bool) -> float:
        """Simple shaped reward; an MPC controller will usually define its own cost."""
        ball_p, ball_v, paddle_p = obs[12:15], obs[15:18], obs[18:21]
        xy_err = np.sum((ball_p[:2] - paddle_p[:2]) ** 2)
        # Predicted apex above the paddle from ballistic flight (drag ignored).
        apex = ball_p[2] - paddle_p[2] + max(ball_v[2], 0.0) ** 2 / (2 * 9.81)
        apex_err = (apex - self.cfg.target_apex) ** 2
        r = 1.0 - 10.0 * xy_err - 5.0 * apex_err + 1.0 * hit
        return float(r - 10.0 * terminated)

    # ------------------------------------------------------------------ for MPC
    def get_state(self) -> np.ndarray:
        state = np.empty(self.nstate)
        mujoco.mj_getState(self.model, self.data, state, FULL_STATE)
        return state

    def set_state(self, state: np.ndarray):
        mujoco.mj_setState(self.model, self.data, np.asarray(state), FULL_STATE)
        mujoco.mj_forward(self.model, self.data)

    def rollout(self, ctrl_seqs: np.ndarray, state: np.ndarray | None = None, nthread: int | None = None):
        """Simulate K open-loop sequences of joint targets in parallel.

        Args:
            ctrl_seqs: (K, H, 6) absolute joint position targets, one per control step.
                Each target is held for ``frame_skip`` physics steps.
            state: initial full-physics state (defaults to the current state).
            nthread: worker threads (defaults to CPU count).

        Returns:
            states: (K, H, nstate) full-physics state at the end of each control step.
            sensors: (K, H, nsensordata), see the <sensor> block in the MJCF
                (paddle_pos, paddle_normal, paddle_linvel, ball_pos, ball_linvel).
        """
        from mujoco import rollout as mj_rollout

        ctrl_seqs = np.asarray(ctrl_seqs, dtype=float)
        if ctrl_seqs.ndim == 2:
            ctrl_seqs = ctrl_seqs[None]
        K, H, _ = ctrl_seqs.shape
        fs = self.cfg.frame_skip
        state0 = self.get_state() if state is None else np.asarray(state)

        if nthread is None:
            import os
            nthread = os.cpu_count() or 1
        while len(self._rollout_data) < nthread:
            self._rollout_data.append(mujoco.MjData(self.model))

        ctrl = np.repeat(np.clip(ctrl_seqs, self.ctrl_low, self.ctrl_high), fs, axis=1)
        # mj_step evaluates sensors *before* integrating, so sensordata[k] describes the
        # state after k steps. One extra physics step makes sensordata line up with the
        # state at the end of every control step (that extra step's state is discarded).
        ctrl = np.concatenate([ctrl, ctrl[:, -1:]], axis=1)
        states, sensors = mj_rollout.rollout(
            self.model, self._rollout_data[:nthread], np.tile(state0, (K, 1)), ctrl
        )
        return states[:, fs - 1:H * fs:fs], sensors[:, fs::fs]

    def sensor_slices(self) -> dict[str, slice]:
        m = self.model
        out = {}
        for i in range(m.nsensor):
            adr, dim = m.sensor_adr[i], m.sensor_dim[i]
            out[m.sensor(i).name] = slice(int(adr), int(adr + dim))
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, width: int = 640, height: int = 480, camera: str = "side") -> np.ndarray:
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
