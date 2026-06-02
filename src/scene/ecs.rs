use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::{Resource, World},
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderObject {
    pub position: [f32; 3],
    pub radius: f32,
    pub object_index: u32,
}

#[derive(Component, Clone, Copy)]
struct SphereObject {
    position: [f32; 3],
    radius: f32,
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

const GRID_DIMENSION: usize = 32;
const GRID_SPACING: f32 = 1.8;
const SPAWN_INTERVAL_SECONDS: f32 = 0.001;
const RADIUS_PATTERN: [f32; 4] = [0.25, 0.40, 0.55, 0.70];

pub(crate) fn build_scene_world() -> SceneWorld {
    let mut world = World::new();
    let center_offset = (GRID_DIMENSION.saturating_sub(1) as f32 * GRID_SPACING) * 0.5;
    let mut entities = Vec::with_capacity(GRID_DIMENSION * GRID_DIMENSION);

    for z in 0..GRID_DIMENSION {
        for x in 0..GRID_DIMENSION {
            let index = entities.len() as u32;
            let radius = RADIUS_PATTERN[index as usize % RADIUS_PATTERN.len()];
            let mut entity = world.spawn(SphereObject {
                position: [
                    x as f32 * GRID_SPACING - center_offset,
                    0.0,
                    z as f32 * GRID_SPACING - center_offset,
                ],
                radius,
                object_index: index,
            });

            if index == 0 {
                entity.insert(Spawned);
            }

            entities.push(entity.id());
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
            radius: sphere.radius,
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
