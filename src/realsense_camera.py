import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps
        )

        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps
        )

        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self.started = False

    def start(self):
        self.pipeline.start(self.config)
        self.started = True

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_bgr = np.asanyarray(color_frame.get_data())
        image_rgb = color_bgr[:, :, ::-1].copy()

        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices())

        cloud = np.zeros((self.height, self.width, 3), dtype=np.float32)

        for i, v in enumerate(vertices):
            y = i // self.width
            x = i % self.width

            cloud[y, x, 0] = v[0]
            cloud[y, x, 1] = v[1]
            cloud[y, x, 2] = v[2]

        return image_rgb, cloud