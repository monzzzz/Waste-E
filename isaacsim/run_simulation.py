"""
Entry point — launch Isaac Sim and run the Waste-E / WPedestrian SIT / Alpamayo simulation.

Run from the Isaac Sim Python environment:

    ./python.sh run_simulation.py
    ./python.sh run_simulation.py --config config.yaml --headless

"""

import argparse
import sys
import os

# Allow imports from this directory when run via Isaac Sim's python.sh
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Waste-E Isaac Sim runner")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (overrides config)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max simulation steps from config",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override pedestrian dataset path from config",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Alpamayo model path from config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Isaac Sim app launcher must be created before any omni imports
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    headless = args.headless or not cfg["simulation"].get("headless", False)
    app_cfg = {
        "headless": headless,
        "physics_gpu": 0,
        "multi_gpu": False,
        "extra_args": ["--/physics/cudaDevice=0"],
    }
    app = SimulationApp(app_cfg)

    # Apply CLI overrides to config
    if args.max_steps is not None:
        cfg["simulation"]["max_steps"] = args.max_steps
    if args.dataset is not None:
        cfg["pedestrians"]["dataset_path"] = args.dataset
    if args.model is not None:
        cfg["alpamayo"]["model_path"] = args.model

    # Save patched config to a temp file so sim_env reads it
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=os.path.dirname(__file__)
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_cfg_path = tmp.name

    try:
        import traceback
        from sim_env import WasteESimEnv
        env = WasteESimEnv(config_path=tmp_cfg_path)
        try:
            env.setup()
            env.run()
        except Exception as e:
            print(f"[run_simulation] CAUGHT EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        except BaseException as e:
            print(f"[run_simulation] CAUGHT BASE EXCEPTION: {type(e).__name__}: {e}", flush=True)
            raise
    finally:
        os.unlink(tmp_cfg_path)
        app.close()


if __name__ == "__main__":
    main()
