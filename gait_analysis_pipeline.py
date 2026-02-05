"""Backward-compatible entrypoint for Colab notebooks.

Some users run `gait_analysis_pipeline.py` from older instructions.
This wrapper forwards execution to the maintained `gait_pipeline.py` CLI.
"""

from gait_pipeline import main


if __name__ == "__main__":
    main()
