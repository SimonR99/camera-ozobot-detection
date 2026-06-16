"""CLI for the robot motion APIs.

Unitree G1 (system Python with unitree_sdk2py):
    /usr/bin/python3 -m motion turn 90        # +90 deg (left/CCW)
    /usr/bin/python3 -m motion turn -90       # right
    /usr/bin/python3 -m motion walk 1.0       # forward 1 m
    /usr/bin/python3 -m motion walk -0.5      # backward
    /usr/bin/python3 -m motion wave           # high wave
    /usr/bin/python3 -m motion wave "face wave"
    /usr/bin/python3 -m motion wave list
    /usr/bin/python3 -m motion stop

LeKiwi (miniforge Python with lerobot):
    python3 -m motion --robot lekiwi turn 45
    python3 -m motion --robot lekiwi turn -45
    python3 -m motion --robot lekiwi walk 0.5
    python3 -m motion --robot lekiwi walk -0.3
    python3 -m motion --robot lekiwi strafe 0.2
    python3 -m motion --robot lekiwi wave
    python3 -m motion --robot lekiwi stop

G1 options: --iface (default eth0), --yes to skip safety prompt.
LeKiwi options: --port (default /dev/ttyACM0), --yes to skip safety prompt.
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motion",
        description="Control the Unitree G1 or LeKiwi robot.",
    )
    p.add_argument(
        "--robot", choices=["g1", "lekiwi"], default="g1",
        help="which robot to control (default: g1)",
    )
    p.add_argument("--yes", action="store_true", help="skip the safety confirmation")

    # G1-specific options
    p.add_argument("--iface", default="eth0", help="G1: network interface (default eth0)")

    # LeKiwi-specific options
    p.add_argument("--port", default="/dev/ttyACM0",
                   help="LeKiwi: serial port (default /dev/ttyACM0)")

    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("turn", help="turn in place by N degrees (+ = left/CCW)")
    t.add_argument("degrees", type=float)
    t.add_argument("--omega", type=float, default=None,
                   help="cruise yaw rate: rad/s for G1, deg/s for LeKiwi")

    w = sub.add_parser("walk", help="walk by N metres (+ = forward)")
    w.add_argument("metres", type=float)
    w.add_argument("--speed", type=float, default=None, help="cruise speed (m/s)")

    s = sub.add_parser("strafe", help="LeKiwi only: strafe by N metres (+ = right)")
    s.add_argument("metres", type=float)
    s.add_argument("--speed", type=float, default=None, help="cruise speed (m/s)")

    g = sub.add_parser("wave", help="perform an arm gesture / wave")
    g.add_argument("gesture", nargs="?", default=None,
                   help="G1: gesture name or 'list'. LeKiwi: --reps N")
    g.add_argument("--reps", type=int, default=2, help="LeKiwi: wave repetitions (default 2)")

    sub.add_parser("stop", help="stop all locomotion")

    return p


def _confirm(action: str) -> bool:
    print(f"WARNING: the robot will {action}. Keep the area clear.")
    try:
        input("Press Enter to continue (Ctrl-C to abort)...")
        return True
    except KeyboardInterrupt:
        print("\nAborted.")
        return False


def _run_g1(args) -> int:
    from .g1_motion import G1Motion, GESTURES

    if args.cmd == "wave" and args.gesture == "list":
        print("Gestures:", ", ".join(GESTURES))
        return 0

    gesture_name = args.gesture or "high wave"
    action_str = {
        "turn": f"turn {getattr(args, 'degrees', '')!s} deg",
        "walk": f"walk {getattr(args, 'metres', '')!s} m",
        "wave": f"perform '{gesture_name}'",
        "stop": "stop",
    }[args.cmd]

    if not args.yes and not _confirm(action_str):
        return 1

    g1 = G1Motion(args.iface)
    if args.cmd == "turn":
        kw = {"omega": args.omega} if args.omega else {}
        print(f"measured = {g1.turn(args.degrees, **kw):+.1f} deg (target {args.degrees:+.0f})")
    elif args.cmd == "walk":
        kw = {"speed": args.speed} if args.speed else {}
        print(f"measured = {g1.walk(args.metres, **kw):+.2f} m (target {args.metres:+.2f})")
    elif args.cmd == "wave":
        print("ExecuteAction return:", g1.wave(gesture_name))
    elif args.cmd == "stop":
        g1.stop()
        print("Stopped.")
    return 0


def _run_lekiwi(args) -> int:
    from .lekiwi_motion import LeKiwiMotion

    action_map = {
        "turn":   f"turn {getattr(args, 'degrees', '')!s} deg",
        "walk":   f"walk {getattr(args, 'metres', '')!s} m",
        "strafe": f"strafe {getattr(args, 'metres', '')!s} m",
        "wave":   "wave arm",
        "stop":   "stop",
    }
    action_str = action_map.get(args.cmd, args.cmd)

    if not args.yes and not _confirm(action_str):
        return 1

    with LeKiwiMotion(args.port) as kiwi:
        if args.cmd == "turn":
            kw = {"omega": args.omega} if args.omega else {}
            kiwi.turn(args.degrees, **kw)
            print(f"Turned {args.degrees:+.0f} deg (open-loop)")
        elif args.cmd == "walk":
            kw = {"speed": args.speed} if args.speed else {}
            kiwi.walk(args.metres, **kw)
            print(f"Walked {args.metres:+.2f} m (open-loop)")
        elif args.cmd == "strafe":
            kw = {"speed": args.speed} if args.speed else {}
            kiwi.strafe(args.metres, **kw)
            print(f"Strafed {args.metres:+.2f} m (open-loop)")
        elif args.cmd == "wave":
            kiwi.wave(reps=args.reps)
            print("Wave complete.")
        elif args.cmd == "stop":
            kiwi.stop()
            print("Stopped.")
    return 0


def main(argv=None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.robot == "g1":
        return _run_g1(args)
    else:
        return _run_lekiwi(args)


if __name__ == "__main__":
    sys.exit(main())
