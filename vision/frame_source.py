"""Frame sources: read BGR frames from an OpenCV camera or a ROS 2 image topic.

Both backends expose the small slice of the ``cv2.VideoCapture`` API the scripts
rely on (``read``/``isOpened``/``release``), so call sites stay backend-agnostic::

    parser = argparse.ArgumentParser()
    add_source_args(parser)
    args = parser.parse_args()
    cap = open_checked(args)          # OpenCV index or ROS topic, depending on args
    ok, frame = cap.read()
    ...
    cap.release()

The ROS backend (``--ros-topic``) subscribes to a ``sensor_msgs/Image`` topic such
as the RealSense colour stream ``/camera/color/image_raw``. ``rclpy`` is
imported lazily so the OpenCV path never requires a ROS install.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np

import cv2

Frame = np.ndarray


class FrameSource:
    """Common interface mirroring the subset of ``cv2.VideoCapture`` we use."""

    def read(self) -> Tuple[bool, Optional[Frame]]:
        raise NotImplementedError

    def isOpened(self) -> bool:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def unavailable_message(self) -> str:
        """Human-readable reason to show when ``isOpened()`` is False."""
        return "Frame source is unavailable"

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class OpenCVCameraSource(FrameSource):
    """Wraps ``cv2.VideoCapture`` for a local ``/dev/video*`` index."""

    def __init__(self, index: int):
        self.index = index
        self._cap = cv2.VideoCapture(index)

    def read(self) -> Tuple[bool, Optional[Frame]]:
        return self._cap.read()

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._cap.release()

    def unavailable_message(self) -> str:
        return (
            f"Cannot open camera {self.index}. On an Intel RealSense the colour "
            f"stream is usually index 4 (0=depth, 2=infrared). The device is also "
            f"'busy' if the realsense2_camera node already holds it — in that case "
            f"use --ros-topic /camera/color/image_raw instead."
        )


def image_msg_to_bgr(msg) -> Frame:
    """Convert a ``sensor_msgs/Image`` into an OpenCV BGR ``ndarray``.

    Handles the encodings the RealSense driver commonly emits (rgb8/bgr8 and the
    alpha/mono variants). Row stride (``msg.step``) padding is respected.
    """
    height, width = msg.height, msg.width
    encoding = (msg.encoding or "").lower()

    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        # Best-effort fallback: infer channels from buffer size.
        total = len(msg.data)
        channels = max(1, total // (height * width)) if height and width else 3

    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = msg.step or width * channels
    arr = buf.reshape(height, step)[:, : width * channels].reshape(height, width, channels)

    if encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr  # bgr8 (or fallback) is already BGR


class Ros2ImageSource(FrameSource):
    """Subscribes to a ``sensor_msgs/Image`` topic and yields the latest frame.

    Uses sensor-data QoS (best-effort) to match the RealSense publisher. ``read()``
    spins the node until a fresh frame arrives or ``timeout_sec`` elapses.
    """

    def __init__(
        self,
        topic: str,
        timeout_sec: float = 5.0,
        node_name: str = "ozobot_frame_source",
    ):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.topic = topic
        self.timeout_sec = timeout_sec
        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._latest: Optional[Frame] = None
        self._sub = self._node.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )

    def _on_image(self, msg) -> None:
        self._latest = image_msg_to_bgr(msg)

    def read(self) -> Tuple[bool, Optional[Frame]]:
        import time

        self._latest = None
        deadline = time.monotonic() + self.timeout_sec
        while self._latest is None and time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        if self._latest is None:
            return False, None
        return True, self._latest

    def isOpened(self) -> bool:
        # Confirm the topic is actually publishing by waiting for one frame.
        ok, _ = self.read()
        return ok

    def release(self) -> None:
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()

    def unavailable_message(self) -> str:
        return (
            f"No frames received on ROS topic '{self.topic}' within "
            f"{self.timeout_sec:.0f}s. Check that the realsense2_camera node is "
            f"running, your ROS 2 workspace is sourced, and the topic name is "
            f"correct (`ros2 topic list`)."
        )


def add_source_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--camera`` / ``--ros-topic`` selection arguments."""
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (RealSense colour stream is usually 4)",
    )
    parser.add_argument(
        "--ros-topic",
        type=str,
        default=None,
        help="Subscribe to a ROS 2 sensor_msgs/Image topic instead of an OpenCV "
        "camera, e.g. /camera/color/image_raw. Takes precedence over --camera.",
    )
    parser.add_argument(
        "--ros-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a ROS image frame before giving up (default: 5)",
    )


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    """Build the frame source selected by ``args`` (ROS topic wins if both given)."""
    topic = getattr(args, "ros_topic", None)
    if topic:
        return Ros2ImageSource(topic, timeout_sec=getattr(args, "ros_timeout", 5.0))
    return OpenCVCameraSource(args.camera)


def open_checked(args: argparse.Namespace) -> FrameSource:
    """Open the selected source and exit with a clear message if it is unavailable."""
    source = open_frame_source(args)
    if not source.isOpened():
        source.release()
        raise SystemExit(source.unavailable_message())
    return source
