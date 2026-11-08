# MPC Ping-Pong Ball Juggling (UR5 + Paddle, MuJoCo)

A MuJoCo environment where a UR5 arm with a table-tennis paddle as its end effector bounces a ping-pong ball.
It is set up for MPC research: it exposes state snapshots and parallel batched rollouts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnltjrqja-dev/cs285/blob/claude/mpc-ping-pong-juggling-k3rhv6/notebooks/colab_demo.ipynb)

## Layout

```
ppjuggle/
  assets/ur5_paddle.xml   # MJCF: UR5 (DH-accurate, primitive geoms) + paddle + ball
  env.py                  # UR5PaddleJuggleEnv: reset/step, get/set_state, rollout, render
scripts/render_demo.py    # renders a drop-test video
notebooks/colab_demo.ipynb
tests/test_env.py         # kinematics, restitution, drag, rollout consistency
```

## Quick start

```bash
pip install -e ".[viz,test]"
pytest -q tests
MUJOCO_GL=egl python scripts/render_demo.py --out demo.mp4   # use MUJOCO_GL=osmesa on CPU-only machines
```

```python
from ppjuggle import UR5PaddleJuggleEnv, EnvConfig, HOME_QPOS
env = UR5PaddleJuggleEnv(EnvConfig(action_mode="absolute"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(HOME_QPOS)

# MPC: K candidate sequences of joint targets, H control steps each
states, sensors = env.rollout(U)        # U: (K, H, 6); env.data is untouched
sl = env.sensor_slices()                # 'paddle_pos', 'paddle_normal', 'paddle_linvel', 'ball_pos', 'ball_linvel'
```

## Model details

| Item | Value | Why |
|---|---|---|
| Arm | UR5 (CB3) DH parameters, real link masses | Kinematics and inertia match the real robot. Links use primitive geoms, so no mesh files are needed. |
| Actuation | Joint position servos (PD), torque limits 150 / 28 Nm, `gravcomp` | Acts like an industrial position controller with gravity feed-forward. |
| Home pose | Paddle center at (0.5, 0, 0.5) m, face pointing +z | Ready to juggle. |
| Paddle | Disc of radius 7.5 cm and thickness 1 cm on the tool flange | Close to a real blade. |
| Ball | 40 mm, 2.7 g, thin-shell inertia | ITTF ball. |
| Contact | `solref="-250000 -52"`, so ω = 500 rad/s and ζ = 0.052 | Restitution about 0.85 and contact time about 6 ms. The result barely changes with timestep. |
| Air | Quadratic drag with Cd ≈ 0.47, Magnus lift | Terminal velocity about 8.5 m/s, as for a real ball. |
| Timestep | 1 ms physics, 10 ms control (`frame_skip=10`) | |

Rough MPC budget: on a CPU, `env.rollout` runs 256 samples × 40-step horizon (102k physics steps) in about 0.4 s.
That is too slow to plan every step in real time. Shorten the horizon, cut the number of samples, or replan only between bounces.
