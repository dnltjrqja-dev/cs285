"""Render a short video of the ball bouncing on the paddle held at the home pose.

    MUJOCO_GL=egl python scripts/render_demo.py --out demo.mp4      # GPU / Colab GPU runtime
    MUJOCO_GL=osmesa python scripts/render_demo.py --out demo.mp4   # CPU only
"""
import argparse

import mediapy
import numpy as np

from ppjuggle import UR5PaddleJuggleEnv

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="demo.mp4")
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--fps", type=int, default=50)
args = parser.parse_args()

env = UR5PaddleJuggleEnv(seed=0)
env.reset()
frames = []
render_every = max(1, round(1 / (args.fps * env.dt)))
for t in range(int(args.seconds / env.dt)):
    _, _, terminated, _, info = env.step(np.zeros(env.act_dim))
    if t % render_every == 0:
        frames.append(env.render())
    if terminated:
        break
mediapy.write_video(args.out, frames, fps=args.fps)
print(f"wrote {args.out}: {len(frames)} frames, hits={info['num_hits']}")
