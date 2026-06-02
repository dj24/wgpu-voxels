use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::{Resource, World},
};

use crate::scene::{OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderObject {
    pub position: [f32; 3],
    pub object_index: u32,
}

#[derive(Component, Clone, Copy)]
struct SphereObject {
    position: [f32; 3],
    object_index: u32,
}

#[derive(Component)]
struct Spawned;

#[derive(Resource)]
struct SpawnProgress {
    next_index: usize,
    interval_seconds: f32,
    accumulated_seconds: f32,
}

#[derive(Resource)]
struct SceneOrder {
    entities: Vec<Entity>,
}

pub(crate) type SceneWorld = World;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActiveSceneSnapshot {
    pub active_count: usize,
}

// Keep the packed scene under the current compute dispatch limit:
// object_count * VOXEL_GRID_DIM / workgroup_size_z must stay <= 65535.
const GRID_DIMENSION: usize = 26;
const GRID_LAYERS: usize = 3;
const SPAWN_INTERVAL_SECONDS: f32 = 0.001;

pub(crate) fn build_scene_world() -> SceneWorld {
    let mut world = World::new();
    let object_extent_x = OBJECT_BOUNDS_MAX[0] - OBJECT_BOUNDS_MIN[0];
    let object_extent_y = OBJECT_BOUNDS_MAX[1] - OBJECT_BOUNDS_MIN[1];
    let object_extent_z = OBJECT_BOUNDS_MAX[2] - OBJECT_BOUNDS_MIN[2];
    let center_offset_x = (GRID_DIMENSION.saturating_sub(1) as f32 * object_extent_x) * 0.5;
    let center_offset_y = (GRID_LAYERS.saturating_sub(1) as f32 * object_extent_y) * 0.5;
    let center_offset_z = (GRID_DIMENSION.saturating_sub(1) as f32 * object_extent_z) * 0.5;
    let mut entities = Vec::with_capacity(GRID_DIMENSION * GRID_DIMENSION * GRID_LAYERS);

    for y in 0..GRID_LAYERS {
        for z in 0..GRID_DIMENSION {
            for x in 0..GRID_DIMENSION {
                let index = entities.len() as u32;
                let mut entity = world.spawn(SphereObject {
                    position: [
                        x as f32 * object_extent_x - center_offset_x,
                        y as f32 * object_extent_y - center_offset_y,
                        z as f32 * object_extent_z - center_offset_z,
                    ],
                    object_index: index,
                });

                if index == 0 {
                    entity.insert(Spawned);
                }

                entities.push(entity.id());
            }
        }
    }

    world.insert_resource(SceneOrder { entities });
    world.insert_resource(SpawnProgress {
        next_index: 1,
        interval_seconds: SPAWN_INTERVAL_SECONDS,
        accumulated_seconds: 0.0,
    });

    world
}

pub(crate) fn advance_spawning(world: &mut SceneWorld, delta_seconds: f32) -> ActiveSceneSnapshot {
    let mut snapshot = snapshot(world);
    if delta_seconds <= 0.0 {
        return snapshot;
    }

    let entity_to_spawn = {
        let (entity_count, entities) = {
            let order = world.resource::<SceneOrder>();
            (order.entities.len(), order.entities.clone())
        };
        let mut progress = world.resource_mut::<SpawnProgress>();
        progress.accumulated_seconds += delta_seconds;

        if progress.accumulated_seconds < progress.interval_seconds
            || progress.next_index >= entity_count
        {
            return snapshot;
        }

        progress.accumulated_seconds -= progress.interval_seconds;
        let entity = entities[progress.next_index];
        progress.next_index += 1;
        entity
    };

    world.entity_mut(entity_to_spawn).insert(Spawned);
    snapshot.active_count += 1;
    snapshot
}

pub(crate) fn collect_all_render_objects(world: &SceneWorld) -> Vec<RenderObject> {
    collect_render_objects(world, true)
}

pub(crate) fn collect_active_render_objects(world: &SceneWorld) -> Vec<RenderObject> {
    collect_render_objects(world, false)
}

fn collect_render_objects(world: &SceneWorld, include_unspawned: bool) -> Vec<RenderObject> {
    let order = &world.resource::<SceneOrder>().entities;
    let mut objects = Vec::with_capacity(order.len());

    for &entity in order {
        let entity_ref = world.entity(entity);
        if !include_unspawned && !entity_ref.contains::<Spawned>() {
            continue;
        }

        let sphere = entity_ref.get::<SphereObject>().expect("sphere object");
        objects.push(RenderObject {
            position: sphere.position,
            object_index: sphere.object_index,
        });
    }

    objects
}

fn snapshot(world: &SceneWorld) -> ActiveSceneSnapshot {
    ActiveSceneSnapshot {
        active_count: collect_active_render_objects(world).len(),
    }
}
