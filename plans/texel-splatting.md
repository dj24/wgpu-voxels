# Texel Splatting
Inspired by https://dylanebert.com/texel-splatting/

To add to the chunk voxel/pixel art aesthetic, we can add texel splatting, which removes shimmering that is usually present by downscaling an image and/or quantising colours.

The original implementation relies on triangle rasterisation into a cubemap probe, which are projected as 1 quad per pixel, scaled up to fill holes.

Another cubemap is rendered directly from the camera to provide the position data for projection of the probe onto the screen, and a fallback image for pixels which sample areas that are occluded from the perspective of the probe.

## Disoclusion Fix

Disoclusion happens if the camera sees a pixel that the probe does not.

As the original implementation uses raster, it only has access to the closest fragment.

As we are raytracing, we can trace past the first hit a few times, to give the projection more "chances" to find a surface.

Although disoclusion is still possible, first the renderer should be tested with no eye cubemap at all, to test if reprojection can be used fully for scene rendering

## Data Storage

We can store multiople cubemaps to contain subsequent ray hits, maybe 3 to start with.

A resolution of 640px can be used for the cubemaps

We can then reproject each cubemap's pixels in sequence - starting from the furthest map, towards the closest.

This will essentially act as furthest-to-closest rendering, by letting each pass of cubemaps splat overwrite the pixels from the previous.

## Quad Splat Dilation

To get a balance between picture quality and minimise gaps between splats, the size of th projected quads (dilation) will need to vary.

This will be achieved by comparing the distance of the sample to the probe, and the eye

| Case                                         | Result                                       |
| -------------------------------------------- | -------------------------------------------- |
| Surface far from probe, also far from camera | Often similar size                           |
| Surface far from probe, near current camera  | Splat becomes large                          |
| Surface near probe, far from current camera  | Splat becomes tiny                           |
| Probe and camera moved sideways              | Gaps can appear around depth discontinuities |

## Algorithm overview

* Trace 3 ray hits via ray query in compute shader, storing gbuffer data in respective cubemap textures
* Reproject the cubemap texels as quads in reverse order (3,2,1), allowing subsequent splat passes to overwrite pixel contents
* Quad sizes derived the splats distance to camera and probe
* Do not have any fallback eye cubemap like the original article - 3 hits should be enough
* For phase 1, we can project all quads as the same size to test, and prevent larger background splats from leaking around foreground splats

## Lighting
Shading can take advantage of the cubemap structure for lighting, as we can detect light occluders via the cubemaps gbuffer inputs and a ray direction.

This does not work too well for indirect lighting, but this could be computed in the shader as an indirect term via an extra ray trace in a cosine-weighted hemispherical sample and combined with cubemap sampled lighting.