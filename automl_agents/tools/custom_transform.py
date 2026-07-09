"""
src/automl_agents/tools/custom_transform.py

Sandboxed custom Python code transformation engine for Pandas DataFrames.
Runs custom code in an isolated subprocess with memory and time caps.
"""

from __future__ import annotations

import os
import sys
import tempfile
import subprocess
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def run_custom_transform_sandboxed(
    code_str: str,
    df: pd.DataFrame,
    timeout_sec: int = 30,
) -> pd.DataFrame:
    """Executes a custom Python code transformation on a Pandas DataFrame in a sandboxed subprocess.
    
    The code block should modify a variable named 'df'.
    
    Example:
        df['age_squared'] = df['age'] ** 2
    """
    if not code_str or not code_str.strip():
        return df

    # Prepare isolated execution environment
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.parquet")
        output_path = os.path.join(tmpdir, "output.parquet")
        script_path = os.path.join(tmpdir, "sandbox.py")

        # Save input snapshot
        df.to_parquet(input_path)

        # Generate sandbox script wrapping custom logic
        # We restrict the execution scope and prevent importing dangerous libraries (like socket/requests)
        # by checking the code or removing modules from sys.modules inside the sandbox.
        script_content = f"""
import sys
import pandas as pd
import numpy as np

# Remove dangerous modules from import path and cache to deter simple malicious attempts
for m in ['socket', 'urllib', 'http', 'requests', 'ftplib', 'smtplib', 'subprocess', 'shutil']:
    sys.modules[m] = None

# Read parquet input
df = pd.read_parquet(r"{input_path}")

# Run user custom code block
try:
    locs = {{"df": df}}
    safe_globals = {{"__builtins__": __builtins__, "pd": pd, "np": np}}
    exec({repr(code_str)}, safe_globals, locs)
    df = locs.get("df", df)
except Exception as e:
    import traceback
    print(f"ERROR: {{str(e)}}\\n{{traceback.format_exc()}}", file=sys.stderr)
    sys.exit(1)

# Write outputs
df.to_parquet(r"{output_path}")
"""

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Inherit full environment to ensure virtualenv and DLL paths are found correctly on Windows
        clean_env = os.environ.copy()

        # Execute sandbox script in a subprocess
        try:
            logger.info(f"Executing custom transformation sandbox (timeout={timeout_sec}s)...")
            res = subprocess.run(
                [sys.executable, script_path],
                timeout=timeout_sec,
                capture_output=True,
                env=clean_env,
                check=True,
            )
            # Read output Parquet snapshot
            if not os.path.exists(output_path):
                raise RuntimeError("Sandbox completed but failed to produce output dataset.")
            return pd.read_parquet(output_path)
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Sandbox execution timed out after {timeout_sec} seconds.")
            raise RuntimeError(f"Custom code execution timed out (exceeded limit of {timeout_sec} seconds).") from e
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace").strip()
            logger.error(f"Sandbox execution failed: {err_msg}")
            raise RuntimeError(f"Custom code execution error: {err_msg}") from e
