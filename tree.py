import taichi as ti
import taichi_glsl as tl
ti.init()

LEAF = -1
kMaxNodes = 512
kMaxParticles = 256

particle_mass = ti.var(ti.f32)
particle_pos = ti.Vector(2, ti.f32)
particle_vel = ti.Vector(2, ti.f32)
particle_table = ti.root.dense(ti.i, kMaxParticles)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
particle_table_len = ti.var(ti.i32, ())

node_mass = ti.var(ti.f32)
node_weighted_pos = ti.Vector(2, ti.f32)
node_particle_id = ti.var(ti.i32)
node_children = ti.var(ti.i32)
node_table = ti.root.dense(ti.i, kMaxNodes)
node_table.place(node_mass, node_particle_id, node_weighted_pos)
node_table.dense(ti.jk, 2).place(node_children)
node_table_len = ti.var(ti.i32, ())

display_image = ti.var(ti.f32, (512, 512))


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
def alloc_a_node_for_particle(particle_id):
    parent = 0
    parent_geo_center = particle_pos[0] * 0 + 0.5
    parent_geo_size = 1.0
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]
    while 1:
        already_have_particle_id = node_particle_id[parent]
        if already_have_particle_id == LEAF:
            break
        node_particle_id[parent] = LEAF

        which_child = abs(position > parent_geo_center)
        child = node_children[parent, which_child]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which_child] = child
        child_geo_center = parent_geo_center + (which_child - 0.5) * parent_geo_size
        child_geo_size = parent_geo_size * 0.5

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

    print(parent)
    node_particle_id[parent] = particle_id
    #node_weighted_pos[parent] = position * mass
    #node_mass[parent] = mass


@ti.kernel
def init():
    node_table_len[None] = 0
    particle_table_len[None] = 0
    alloc_node()


@ti.kernel
def render(mx: ti.f32, my: ti.f32):
    for pixel in ti.grouped(display_image):
        display_image[pixel] = 0

    mouse_pos = tl.vec(mx, my)

    particle_id = alloc_particle()
    particle_pos[particle_id] = mouse_pos
    alloc_a_node_for_particle(particle_id)

    for which in ti.static(ti.grouped(ti.ndrange(2, 2))):
        child = node_children[0, which]
        for pixel in ti.grouped(ti.ndrange(256, 256)):
            display_image[which * 256 + pixel] = tl.mix(0, 0.1, child != LEAF)


init()
gui = ti.GUI('Tree-code')
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            render(*gui.get_cursor_pos())
    gui.set_image(display_image)
    gui.show()
