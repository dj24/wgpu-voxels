# interpolation pass

As an experiment, we could use our existing render targets and run a post process with smooth shading.

By comparing a world position reconstructed from depth against the stored world position (per voxel world pos), we essentially get a 3d uv of a fragment within a voxel.

Then, by using a a-trous filter (perhaps dependent on voxel screen size), we search to find neighbouring voxels.

Finally, blend between the found neighbours based on the voxel uv.

For an added step, we could try using bicubic filtering for smoother transitions.

This would output to a new texture, as we want to keep all the existing per voxel shading in tact.

Add this to number 8 hotkey to see it