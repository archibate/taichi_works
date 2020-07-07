import taichi as ti
import taichi_glsl as tl
ti.init()

LEAF = -1
TREE = -2
kMaxNodes = 512
kMaxParticles = 256

particle_mass = ti.var(ti.f32)
particle_pos = ti.Vector(2, ti.f32)
particle_vel = ti.Vector(2, ti.f32)
particle_table = ti.root.dense(ti.i, kMaxParticles)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
particle_table_len = ti.var(ti.i32, ())

trash_particle_id = ti.var(ti.i32)
trash_table = ti.root.dense(ti.i, kMaxParticles)
trash_table.place(trash_particle_id)
trash_table_len = ti.var(ti.i32, ())

node_mass = ti.var(ti.f32)
node_weighted_pos = ti.Vector(2, ti.f32)
node_particle_id = ti.var(ti.i32)
node_children = ti.var(ti.i32)
node_table = ti.root.dense(ti.i, kMaxNodes)
node_table.place(node_mass, node_particle_id, node_weighted_pos)
node_table.dense(ti.jk, 2).place(node_children)
node_table_len = ti.var(ti.i32, ())

display_image = ti.Vector(3, ti.f32, (512, 512))


@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < kMaxNodes
    node_mass[ret] = 0
    node_weighted_pos[ret] = 0
    node_particle_id[ret] = LEAF
    for which in ti.grouped(ti.ndrange(2, 2)):
        node_children[ret, which] = LEAF
    return ret


@ti.func
def alloc_particle():
    ret = ti.atomic_add(particle_table_len[None], 1)
    assert ret < kMaxParticles
    particle_mass[ret] = 0
    particle_pos[ret] = 0
    particle_vel[ret] = 0
    return ret


@ti.func
def append_trash(particle_id):
    ret = ti.atomic_add(trash_table_len[None], 1)
    assert ret < kMaxParticles
    trash_particle_id[ret] = particle_id
    return ret


@ti.func
def alloc_a_node_for_particle(particle_id):
    parent = 0
    parent_geo_center = particle_pos[0] * 0 + 0.5
    parent_geo_size = 1.0
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]
    while 1:
        already_particle_id = node_particle_id[parent]
        if already_particle_id == LEAF:
            break
        if already_particle_id != TREE:
            node_particle_id[parent] = TREE
            append_trash(already_particle_id)

        which_child = abs(position > parent_geo_center)
        child = node_children[parent, which_child]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which_child] = child
        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which_child - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

    node_particle_id[parent] = particle_id
    node_weighted_pos[parent] = position * mass
    node_mass[parent] = mass


@ti.kernel
def add_particle_at(mx: ti.f32, my: ti.f32, mass: ti.f32):
    mouse_pos = tl.vec(mx, my)

    particle_id = alloc_particle()
    particle_pos[particle_id] = mouse_pos
    particle_mass[particle_id] = mass


@ti.kernel
def build_tree():
    node_table_len[None] = 0
    trash_table_len[None] = 0
    alloc_node()

    particle_id = 0
    while particle_id < particle_table_len[None]:
        alloc_a_node_for_particle(particle_id)

        trash_id = 0
        while trash_id < trash_table_len[None]:
            alloc_a_node_for_particle(trash_particle_id[trash_id])
            trash_id = trash_id + 1

        trash_table_len[None] = 0
        particle_id = particle_id + 1


@ti.func
def get_raw_gravity_at(pos):
    acc = particle_pos[0] * 0
    for i in particle_pos:
        acc += particle_mass[i] * tl.normalizePow(particle_pos[i] - pos, -2, 1e-3)
    return acc


@ti.kernel
def render_arrows():
    particle_id = max(0, particle_table_len[None] - 1)
    pos = particle_pos[particle_id]
    acc = get_raw_gravity_at(pos) * 0.001
    tl.paintArrow(display_image, pos, acc, tl.D.yxy)


def render_tree(gui, parent=0, parent_geo_center=tl.vec(0.5, 0.5), parent_geo_size=1.0):
    child_geo_size = parent_geo_size * 0.5
    if node_particle_id[parent] >= 0:
        tl = parent_geo_center - child_geo_size
        br = parent_geo_center + child_geo_size
        gui.rect(tl, br, radius=1, color=0xff0000)
    for which in map(ti.Vector, [[0, 0], [0, 1], [1, 0], [1, 1]]):
        child = node_children[(parent, which[0], which[1])]
        if child >= 0:
            tl = parent_geo_center + (which - 1) * child_geo_size
            br = parent_geo_center + which * child_geo_size
            child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size
            gui.rect(tl, br, radius=1, color=0xff0000)
            render_tree(gui, child, child_geo_center, child_geo_size)


gui = ti.GUI('Tree-code')
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key in [gui.LMB, gui.RMB]:
            add_particle_at(*gui.get_cursor_pos(), e.key == gui.LMB)
    build_tree()
    display_image.fill(0)
    render_arrows()
    gui.set_image(display_image)
    render_tree(gui)
    gui.circles(particle_pos.to_numpy()[:particle_table_len[None]], radius=3)
    gui.show()
