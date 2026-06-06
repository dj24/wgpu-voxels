use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::{Resource, World},
};

use crate::scene::{OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN};

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VoxelGenerationKind {
    Terrain,
    Cornell,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderObject {
    pub position: [f32; 3],
    pub object_index: u32,
    pub generation_kind: VoxelGenerationKind,
}

#[derive(Component, Clone, Copy)]
struct ChunkObject {
    position: [f32; 3],
    object_index: u32,
    generation_kind: VoxelGenerationKind,
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

#[derive(Clone, Copy)]
struct ChunkDescriptor {
    position: [f32; 3],
    object_index: u32,
    generation_kind: VoxelGenerationKind,
    initially_loaded: bool,
}

pub(crate) type SceneWorld = World;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActiveSceneSnapshot {
    pub active_count: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
const GRID_DIMENSION: usize = 26;
#[cfg_attr(not(test), allow(dead_code))]
const GRID_LAYERS: usize = 3;
const MAX_ACTIVE_CHUNKS: usize = 128;

pub(crate) fn build_scene_world() -> SceneWorld {
    let mut world = World::new();
    let descriptors = build_cornell_chunk_descriptors();
    spawn_chunk_descriptors(&mut world, &descriptors);
    world
}

fn spawn_chunk_descriptors(world: &mut SceneWorld, descriptors: &[ChunkDescriptor]) {
    let mut entities = Vec::with_capacity(descriptors.len());
    let mut initially_loaded_count = 0usize;

    for descriptor in descriptors {
        let mut entity = world.spawn(ChunkObject {
            position: descriptor.position,
            object_index: descriptor.object_index,
            generation_kind: descriptor.generation_kind,
        });

        if descriptor.initially_loaded {
            entity.insert(Loaded);
            initially_loaded_count += 1;
        }

        entities.push(entity.id());
    }

    world.insert_resource(SceneOrder { entities });
    world.insert_resource(LoadProgress {
        next_index: initially_loaded_count,
    });
}

#[cfg_attr(not(test), allow(dead_code))]
fn build_terrain_chunk_descriptors() -> Vec<ChunkDescriptor> {
    let object_extent_x = OBJECT_BOUNDS_MAX[0] - OBJECT_BOUNDS_MIN[0];
    let object_extent_y = OBJECT_BOUNDS_MAX[1] - OBJECT_BOUNDS_MIN[1];
    let object_extent_z = OBJECT_BOUNDS_MAX[2] - OBJECT_BOUNDS_MIN[2];
    let center_offset_x = (GRID_DIMENSION.saturating_sub(1) as f32 * object_extent_x) * 0.5;
    let center_offset_y = (GRID_LAYERS.saturating_sub(1) as f32 * object_extent_y) * 0.5;
    let center_offset_z = (GRID_DIMENSION.saturating_sub(1) as f32 * object_extent_z) * 0.5;
    let mut descriptors = Vec::with_capacity(GRID_DIMENSION * GRID_DIMENSION * GRID_LAYERS);

    for y in 0..GRID_LAYERS {
        for z in 0..GRID_DIMENSION {
            for x in 0..GRID_DIMENSION {
                let object_index = descriptors.len() as u32;
                descriptors.push(ChunkDescriptor {
                    position: [
                        x as f32 * object_extent_x - center_offset_x,
                        y as f32 * object_extent_y - center_offset_y,
                        z as f32 * object_extent_z - center_offset_z,
                    ],
                    object_index,
                    generation_kind: VoxelGenerationKind::Terrain,
                    initially_loaded: object_index == 0,
                });
            }
        }
    }

    descriptors
}

fn build_cornell_chunk_descriptors() -> Vec<ChunkDescriptor> {
    vec![ChunkDescriptor {
        position: [0.0, 0.0, 0.0],
        object_index: 0,
        generation_kind: VoxelGenerationKind::Cornell,
        initially_loaded: true,
    }]
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
            generation_kind: chunk.generation_kind,
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
        ActiveSceneSnapshot, GRID_DIMENSION, GRID_LAYERS, VoxelGenerationKind,
        advance_chunk_loading, build_cornell_chunk_descriptors, build_scene_world,
        build_terrain_chunk_descriptors, collect_active_render_objects,
    };

    #[test]
    fn terrain_helper_preserves_grid_shape() {
        let descriptors = build_terrain_chunk_descriptors();

        assert_eq!(
            descriptors.len(),
            GRID_DIMENSION * GRID_DIMENSION * GRID_LAYERS
        );
        assert_eq!(descriptors[0].object_index, 0);
        assert_eq!(
            descriptors.last().expect("terrain descriptor").object_index as usize,
            descriptors.len() - 1
        );
        assert_eq!(
            descriptors
                .iter()
                .filter(|descriptor| descriptor.initially_loaded)
                .count(),
            1
        );
        assert!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.generation_kind == VoxelGenerationKind::Terrain)
        );
    }

    #[test]
    fn default_world_starts_with_loaded_cornell_scene() {
        let world = build_scene_world();
        let active_objects = collect_active_render_objects(&world);

        assert_eq!(active_objects.len(), 1);
        assert_eq!(active_objects[0].object_index, 0);
        assert_eq!(
            active_objects[0].generation_kind,
            VoxelGenerationKind::Cornell
        );
    }

    #[test]
    fn cornell_descriptors_are_loaded_and_contiguous() {
        let descriptors = build_cornell_chunk_descriptors();

        assert_eq!(descriptors.len(), 1);
        assert!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.initially_loaded)
        );
        assert_eq!(descriptors[0].object_index, 0);
    }

    #[test]
    fn fully_loaded_cornell_scene_does_not_stream_more_chunks() {
        let mut world = build_scene_world();

        let first = advance_chunk_loading(&mut world);
        let second = advance_chunk_loading(&mut world);

        assert_eq!(first, ActiveSceneSnapshot { active_count: 1 });
        assert_eq!(second, ActiveSceneSnapshot { active_count: 1 });
        assert_eq!(collect_active_render_objects(&world).len(), 1);
    }
}
