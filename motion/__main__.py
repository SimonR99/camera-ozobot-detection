"""CLI for the G1 motion API.

    /usr/bin/python3 -m motion turn 90        # +90 deg (left)
    /usr/bin/python3 -m motion turn -90       # right
    /usr/bin/python3 -m motion walk 1.0       # forward 1 m
    /usr/bin/python3 -m motion walk -0.5      # backward
    /usr/bin/python3 -m motion wave           # high wave
    /usr/bin/python3 -m motion wave "face wave"
    /usr/bin/python3 -m motion wave list      # list gestures
    /usr/bin/python3 -m motion stop

Use --iface to pick the interface (default eth0), --yes to skip the safety prompt.
"""

import argparse
import sys

from .g1_motion import G1Motion, GESTURES


def main(argv=None):
    p = argparse.ArgumentParser(prog="motion", description="Control the Unitree G1.")
    p.add_argument("--iface", default="eth0", help="network interface to the robot")
    p.add_argument("--yes", action="store_true", help="skip the safety confirmation")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("turn", help="turn in place by N degrees (+ = left)")
    t.add_argument("degrees", type=float)
    t.add_argument("--omega", type=float, default=None, help="cruise yaw rate (rad/s)")

    w = sub.add_parser("walk", help="walk straight by N metres (+ = forward)")
    w.add_argument("metres", type=float)
    w.add_argument("--speed", type=float, default=None, help="cruise speed (m/s)")

    g = sub.add_parser("wave", help="perform an arm gesture")
    g.add_argument("gesture", nargs="?", default="high wave",
                   help="gesture name, or 'list' to show all")

    sub.add_parser("stop", help="stop all locomotion")

    args = p.parse_args(argv)

    if args.cmd == "wave" and args.gesture == "list":
        print("Gestures:", ", ".join(GESTURES))
        return 0

    if not args.yes:
        action = {
            "turn": f"turn {getattr(args, 'degrees', '')!s} deg",
            "walk": f"walk {getattr(args, 'metres', '')!s} m",
            "wave": f"perform '{getattr(args, 'gesture', '')}'",
            "stop": "stop",
        }[args.cmd]
        print(f"WARNING: the robot will {action}. Keep the area clear, E-stop in reach.")
        try:
            input("Press Enter to continue (Ctrl-C to abort)...")
        except KeyboardInterrupt:
            print("\nAborted.")
            return 1

    g1 = G1Motion(args.iface)
    if args.cmd == "turn":
        kw = {"omega": args.omega} if args.omega else {}
        print(f"measured = {g1.turn(args.degrees, **kw):+.1f} deg (target {args.degrees:+.0f})")
    elif args.cmd == "walk":
        kw = {"speed": args.speed} if args.speed else {}
        print(f"measured = {g1.walk(args.metres, **kw):+.2f} m (target {args.metres:+.2f})")
    elif args.cmd == "wave":
        print("ExecuteAction return:", g1.wave(args.gesture))
    elif args.cmd == "stop":
        g1.stop()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
