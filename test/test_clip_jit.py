import subprocess
import sys
for i in range(8): subprocess.run([sys.executable, "test/run_clip.py", str(2**i)], check=True)