import os

logdir = "notebooks/outputs/deep_fib/tb"

try:
    os.system(f"tensorboard --logdir={logdir}")
except KeyboardInterrupt:
    print("Closing Session")
