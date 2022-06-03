import os

logdir = "notebooks/deep_fib_outs/tb"

try:
    os.system(f"tensorboard --logdir={logdir}")
except KeyboardInterrupt:
    print("Closing Session")
