import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

realsense_ctx = rs.context()
connected_devices = []
configs = []
pipelines = []

for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    connected_devices.append(detected_camera)
    configs.append(rs.config())
    pipelines.append(rs.pipeline())

print(connected_devices)

for i in range(len(configs)):
    configs[i].enable_device(connected_devices[i])
    configs[i].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    configs[i].enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipelines[i].start(configs[i])
    profile = pipelines[i].get_active_profile()

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w=depth_intrinsics.width
    h=depth_intrinsics.height

pc = []
# Processing blocks
for i in range(len(pipelines)):
    pc.append(rs.pointcloud())
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()
    
def project(out,v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(out,pt1.reshape(-1, 3))[0]
    p1 = project(out,pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(out,v[s])
    else:
        proj = project(out,view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

outs = []


while True:

    for i in range(len(pipelines)):
        outs.append(np.empty((h, w, 3), dtype=np.uint8))
    # Grab camera data
    frames = []
    depth_frames = []
    color_frames = []
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        for i in range(len(pipelines)):
            frames.append(pipelines[i].wait_for_frames())
        for i in range(len(pipelines)):
            depth_frames.append(frames[i].get_depth_frame())
            color_frames.append(frames[i].get_color_frame())
            depth_frames[i] = decimate.process(depth_frames[i])

        # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
            depth_frames[i].profile).get_intrinsics()               #bacha!!!!
            w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_images = []
        color_images = []
        depth_colormaps = []
        mapped_frames = []
        color_sources = []
        for i in range(len(pipelines)):
            depth_images.append(np.asanyarray(depth_frames[i].get_data()))
            color_images.append(np.asanyarray(color_frames[i].get_data()))

            depth_colormaps.append(np.asanyarray(
            colorizer.colorize(depth_frames[i]).get_data()))

            if state.color:
                mapped_frames.append(color_frames[i])
                color_sources.append(color_images[i])
            else:
                mapped_frames.append(depth_frames[i])
                color_sources.append(depth_colormaps[i])

        points = []
        verts = []
        texcoords = []
        for i in range(len(pipelines)):
            points.append(pc[i].calculate(depth_frames[i]))
            pc[i].map_to(mapped_frames[i])

        # Pointcloud data to arrays
            v, t = points[i].get_vertices(), points[i].get_texture_coordinates()
            verts.append(np.asanyarray(v).view(np.float32).reshape(-1, 3))  # xyz
            texcoords.append(np.asanyarray(t).view(np.float32).reshape(-1, 2))  # uv

        frames.clear()
        depth_frames.clear()
        color_frames.clear()
        depth_images.clear()
        color_images.clear()
        depth_colormaps.clear()
        
        mapped_frames.clear()
    # Render
    now = time.time()

    for i in range(len(pipelines)):
        outs[i].fill(0)

        grid(outs[i], (0, 0.5, 1), size=1, n=10)
        frustum(outs[i], depth_intrinsics)
        axes(outs[i], view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or outs[i].shape[:2] == (h, w):
            pointcloud(outs[i], verts[i], texcoords[i], color_sources[i])
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts[i], texcoords[i], color_sources[i])
            tmp = cv2.resize(tmp, outs[i].shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(outs[i], tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(outs[i], view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    for i in range(len(pipelines)):
        cv2.imshow(state.WIN_NAME + str(i), outs[i])
        key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

    outs.clear()
    verts.clear()
    texcoords.clear()
    color_sources.clear()

for pipeline in pipelines:
    pipeline.stop()