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
struct ChunkObject {
    position: [f32; 3],
    object_index: u32,
}

#[derive(Component)]
struct Loaded;

#[derive(Resource)]
struct SceneOrder {
    entities: Vec<Entity>,
}

#[derive(Resource)]
struct LoadProgress {
    next_index: usize,
}

pub(crate) type SceneWorld = World;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActiveSceneSnapshot {
    pub active_count: usize,
}

const GRID_DIMENSION: usize = 26;
const GRID_LAYERS: usize = 3;
const MAX_ACTIVE_CHUNKS: usize = 128;

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
                let mut entity = world.spawn(ChunkObject {
                    position: [
                        x as f32 * object_extent_x - center_offset_x,
                        y as f32 * object_extent_y - center_offset_y,
                        z as f32 * object_extent_z - center_offset_z,
                    ],
                    object_index: index,
                });

                if index == 0 {
                    entity.insert(Loaded);
                }

                entities.push(entity.id());
            }
        }
    }

    world.insert_resource(SceneOrder { entities });
    world.insert_resource(LoadProgress { next_index: 1 });
    world
}

pub(crate) fn collect_active_render_objects(world: &SceneWorld) -> Vec<RenderObject> {
    collect_render_objects(world, false)
}

pub(crate) fn advance_chunk_loading(world: &mut SceneWorld) -> ActiveSceneSnapshot {
    let entities = world.resource::<SceneOrder>().entities.clone();
    let total_count = entities.len();
    let next_entity = {
        let mut progress = world.resource_mut::<LoadProgress>();
        if progress.next_index >= total_count || progress.next_index >= MAX_ACTIVE_CHUNKS {
            return snapshot(world);
        }

        let entity = entities[progress.next_index];
        progress.next_index += 1;
        entity
    };

    world.entity_mut(next_entity).insert(Loaded);
    ActiveSceneSnapshot {
        active_count: world.resource::<LoadProgress>().next_index,
    }
}

pub(crate) fn load_max_active_chunks(world: &mut SceneWorld) -> ActiveSceneSnapshot {
    let mut previous = snapshot(world);
    loop {
        let current = advance_chunk_loading(world);
        if current == previous {
            return current;
        }
        previous = current;
    }
}

fn collect_render_objects(world: &SceneWorld, include_unloaded: bool) -> Vec<RenderObject> {
    let order = &world.resource::<SceneOrder>().entities;
    let mut objects = Vec::with_capacity(order.len());

    for &entity in order {
        let entity_ref = world.entity(entity);
        if !include_unloaded && !entity_ref.contains::<Loaded>() {
            continue;
        }

        let chunk = entity_ref.get::<ChunkObject>().expect("chunk object");
        objects.push(RenderObject {
            position: chunk.position,
            object_index: chunk.object_index,
        });
    }

    objects
}

fn snapshot(world: &SceneWorld) -> ActiveSceneSnapshot {
    ActiveSceneSnapshot {
        active_count: collect_active_render_objects(world).len(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActiveSceneSnapshot, advance_chunk_loading, build_scene_world,
        collect_active_render_objects,
    };

    #[test]
    fn world_starts_with_one_loaded_chunk() {
        let world = build_scene_world();
        let active_objects = collect_active_render_objects(&world);

        assert_eq!(active_objects.len(), 1);
        assert_eq!(active_objects[0].object_index, 0);
    }

    #[test]
    fn loading_advances_one_chunk_at_a_time() {
        let mut world = build_scene_world();

        let first = advance_chunk_loading(&mut world);
        let second = advance_chunk_loading(&mut world);

        assert_eq!(first, ActiveSceneSnapshot { active_count: 2 });
        assert_eq!(second, ActiveSceneSnapshot { active_count: 3 });
        assert_eq!(collect_active_render_objects(&world).len(), 3);
    }
}
