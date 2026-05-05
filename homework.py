import math
import sys

try:
    import taichi as ti
except ImportError as exc:
    raise SystemExit(
        "Taichi is not installed. Please run: pip install taichi\n"
        "Then run: python whitted_ray_tracing_taichi.py"
    ) from exc

WIDTH = 960
HEIGHT = 540
ASPECT = WIDTH / HEIGHT
FOV_DEGREES = 45.0
MAX_BOUNCES_LIMIT = 5
EPSILON = 1.0e-4
INF = 1.0e20

MAT_DIFFUSE = 0
MAT_MIRROR = 1

try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
except Exception:
    print("[Warning] GPU backend unavailable; falling back to CPU.")
    ti.init(arch=ti.cpu, default_fp=ti.f32)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())


@ti.func
def normalize_safe(v):
    length = ti.sqrt(v.dot(v))
    out = ti.Vector([0.0, 0.0, 0.0])
    if length > 1.0e-8:
        out = v / length
    return out


@ti.func
def reflect(incident, normal):
    return normalize_safe(incident - 2.0 * incident.dot(normal) * normal)


@ti.func
def sphere_intersect(ray_o, ray_d, center, radius):
    oc = ray_o - center
    b = oc.dot(ray_d)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - c
    t = INF
    if discriminant > 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t0 = -b - sqrt_disc
        t1 = -b + sqrt_disc
        if t0 > EPSILON:
            t = t0
        elif t1 > EPSILON:
            t = t1
    return t


@ti.func
def intersect_scene(ray_o, ray_d):
    hit = 0
    closest_t = INF
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_n = ti.Vector([0.0, 1.0, 0.0])
    albedo = ti.Vector([0.0, 0.0, 0.0])
    material_id = MAT_DIFFUSE

    red_center = ti.Vector([-1.5, 0.0, 0.0])
    t_red = sphere_intersect(ray_o, ray_d, red_center, 1.0)
    if t_red < closest_t:
        hit = 1
        closest_t = t_red
        hit_p = ray_o + closest_t * ray_d
        hit_n = normalize_safe(hit_p - red_center)
        albedo = ti.Vector([0.95, 0.08, 0.05])
        material_id = MAT_DIFFUSE

    mirror_center = ti.Vector([1.5, 0.0, 0.0])
    t_mirror = sphere_intersect(ray_o, ray_d, mirror_center, 1.0)
    if t_mirror < closest_t:
        hit = 1
        closest_t = t_mirror
        hit_p = ray_o + closest_t * ray_d
        hit_n = normalize_safe(hit_p - mirror_center)
        albedo = ti.Vector([0.88, 0.88, 0.90])
        material_id = MAT_MIRROR

    if ti.abs(ray_d.y) > 1.0e-6:
        t_plane = (-1.0 - ray_o.y) / ray_d.y
        if t_plane > EPSILON and t_plane < closest_t:
            hit = 1
            closest_t = t_plane
            hit_p = ray_o + closest_t * ray_d
            hit_n = ti.Vector([0.0, 1.0, 0.0])
            ix = ti.cast(ti.floor(hit_p.x), ti.i32)
            iz = ti.cast(ti.floor(hit_p.z), ti.i32)
            checker = (ix + iz) & 1
            if checker == 0:
                albedo = ti.Vector([0.88, 0.88, 0.88])
            else:
                albedo = ti.Vector([0.08, 0.08, 0.08])
            material_id = MAT_DIFFUSE

    return hit, closest_t, hit_p, hit_n, albedo, material_id


@ti.func
def is_shadowed(point, direction_to_light, max_dist):
    hit, t, hp, hn, alb, mat = intersect_scene(point, direction_to_light)
    blocked = 0
    if hit == 1 and t < max_dist - EPSILON:
        blocked = 1
    return blocked


@ti.func
def background_color(ray_d):
    t = 0.5 * (ray_d.y + 1.0)
    return (1.0 - t) * ti.Vector([0.78, 0.82, 0.90]) + t * ti.Vector([0.20, 0.32, 0.55])


@ti.kernel
def render():
    camera_pos = ti.Vector([0.0, 1.0, 5.5])
    camera_target = ti.Vector([0.0, 0.0, 0.0])
    camera_up_hint = ti.Vector([0.0, 1.0, 0.0])

    forward = normalize_safe(camera_target - camera_pos)
    right = normalize_safe(forward.cross(camera_up_hint))
    up = normalize_safe(right.cross(forward))
    tan_half_fov = ti.tan((FOV_DEGREES * math.pi / 180.0) * 0.5)

    for i, j in pixels:
        u = (2.0 * ((ti.cast(i, ti.f32) + 0.5) / WIDTH) - 1.0) * ASPECT * tan_half_fov
        v = (2.0 * ((ti.cast(j, ti.f32) + 0.5) / HEIGHT) - 1.0) * tan_half_fov

        ray_o = camera_pos
        ray_d = normalize_safe(forward + u * right + v * up)

        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])

        for bounce in range(MAX_BOUNCES_LIMIT):
            if bounce >= max_bounces[None]:
                break

            hit, t_hit, p, n, albedo, material_id = intersect_scene(ray_o, ray_d)

            if hit == 0:
                final_color += throughput * background_color(ray_d)
                break

            if n.dot(ray_d) > 0.0:
                n = -n

            if material_id == MAT_MIRROR:
                ray_o = p + n * EPSILON
                ray_d = reflect(ray_d, n)
                throughput *= albedo * 0.8
            else:
                lp = light_pos[None]
                to_light_vec = lp - p
                light_distance = ti.sqrt(to_light_vec.dot(to_light_vec))
                to_light_dir = normalize_safe(to_light_vec)

                ambient = 0.12
                diffuse = 0.0
                if light_distance > EPSILON:
                    blocked = is_shadowed(p + n * EPSILON, to_light_dir, light_distance)
                    if blocked == 0:
                        diffuse = ti.max(n.dot(to_light_dir), 0.0)

                light_color = ti.Vector([1.0, 0.96, 0.88])
                direct_light_strength = 1.25
                shaded = albedo * (ambient + direct_light_strength * diffuse * light_color)
                final_color += throughput * shaded
                break

        final_color = ti.min(ti.max(final_color, 0.0), 1.0)
        pixels[i, j] = ti.sqrt(final_color)


def main():
    light_x = 0.0
    light_y = 4.0
    light_z = 3.0
    bounce_count = 3

    light_pos[None] = [light_x, light_y, light_z]
    max_bounces[None] = bounce_count

    window = ti.ui.Window("Whitted-Style Ray Tracing - Hard Shadows and Mirror Reflection", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.26):
            gui.text("Distance unit: SU (Scene Unit)")
            gui.text("Plane y = -1.0 SU; sphere radius = 1.0 SU")
            light_x = gui.slider_float("Light X (SU)", light_x, -5.0, 5.0)
            light_y = gui.slider_float("Light Y (SU)", light_y, 0.2, 8.0)
            light_z = gui.slider_float("Light Z (SU)", light_z, -5.0, 6.0)
            bounce_count = gui.slider_int("Max Bounces", bounce_count, 1, MAX_BOUNCES_LIMIT)
            gui.text("1 bounce: no mirror world; >1: reflected world appears")
            gui.text("Shadow acne fix: P_new = P + N * 1e-4 SU")

        light_pos[None] = [light_x, light_y, light_z]
        max_bounces[None] = bounce_count
        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
