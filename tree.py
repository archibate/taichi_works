import taichi as ti
import taichi_glsl as tl
ti.init()

kMaxNodes = 512
kMaxParticles = 256

particle_mass = ti.var(ti.f32)
particle_pos = ti.Vector(2, ti.f32)
particle_vel = ti.Vector(2, ti.f32)
particle_table = ti.root.dense(ti.i, kMaxParticles)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
particle_table_len = ti.var(ti.i32, ())

node_mass = ti.var(ti.f32)
node_center = ti.Vector(2, ti.f32)
node_particle_id = ti.var(ti.i32)
node_children = ti.Matrix(2, 2, ti.i32)
node_table = ti.root.dense(ti.i, kMaxNodes)
node_table.place(node_mass, node_particle_id, node_center).place(node_children)
node_table_len = ti.var(ti.i32, ())


@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < kMaxNodes
    node_mass[ret] = 0
    node_center[ret] = 0
    node_particle_id[ret] = 0
    node_children[ret] = 0
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
def alloc_a_node_for_position(position):
    parent = 0
    parent_geo_center = particle_pos[0] * 0 + 0.5
    parent_geo_size = 1
    while 1:
        which_child = abs(position > parent_geo_center)
        child = node_children[parent][which_child]
        if child == 0:
            child = alloc_node()
            node_children[parent][which_child] = child
        child_geo_center = parent_geo_center + (which_child - 0.5) * parent_geo_size
        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child
