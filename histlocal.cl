#define MAX_LOCAL_SIZE 512

__kernel void addToHistogramLocal(
    __global PyroXY* zin,
    __global PyroRGB*  colour,
    cfloat_t minxy, // the minimum (x,y) point captured by the histogram.
    cfloat_t maxxy, // the max (x,y) point captured by the histogram.
    __global PyroHistogramRGB* histogram,
    int image_width,
    float pixel_per_unit)
{
  int gid = get_global_id(0); // The x-coordinate that we are writing to the histogram.
  int lid = get_local_id(0);

  float x = zin[gid].x;
  float y = -zin[gid].y; // mirror y

  __local int indices[MAX_LOCAL_SIZE]; // short cut: There are no more than 512 local work items
  __local HistogramCache cache[MAX_LOCAL_SIZE];

  int histogram_index = -1;
  int mincacheid = lid; // the minimum local id with the same index into histogram. Initialize to the curent local id.

  if (x >= minxy.x && x < maxxy.x && y >= minxy.y && y < maxxy.y)
  {
    int pixel_x = floor((x - minxy.x) * pixel_per_unit);
    int pixel_y = floor((y - minxy.y) * pixel_per_unit);

    // We need to plot into the histogram at element histogram_index
    histogram_index = pixel_y * image_width + pixel_x;
  }

  // Keep track of which local IDs need to write into which histogrram index.
  indices[lid] = histogram_index;

  if (histogram_index > 0)
  {
    // Wait until all work items in the local group have set their index in indices
    barrier(CLK_LOCAL_MEM_FENCE);

    // This local work item needs to plot into the index histogram_index. Find the lowest local ID
    // that also needs to plot into this index.
    for (int otherid = lid-1; otherid >= 0; --otherid)
    {
        if (indices[otherid] == histogram_index)
        {
          mincacheid = otherid;
        }
    }

    // Initialize the local caches that are owned by the minimum local ID work items that are writing to the histogram.
    if (lid == mincacheid)
    {
      // Initialize all the cache entries to zero.
      cache[lid].r = 0.0;
      cache[lid].g = 0.0;
      cache[lid].b = 0.0;
      cache[lid].count = 0.0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work item needs to accumulate its colour and count into the cache entry owned by the minimum local ID that is
    // writing into the samw histogram index.
    pyro_atomic_add_float_local((__local float*) &cache[mincacheid].r, colour[gid].r);
    pyro_atomic_add_float_local((__local float*) &cache[mincacheid].g, colour[gid].g);
    pyro_atomic_add_float_local((__local float*) &cache[mincacheid].b, colour[gid].b);
    pyro_atomic_add_float_local((__local float*) &cache[mincacheid].count, 1.0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // If this is the minimum local ID writing to any index, perform a bulk read of the whole histogram bin, increment
    // and write the whole bin back to global memory.
    if (lid == mincacheid)
    {
      PyroHistogramRGB h = histogram[histogram_index];
      h.r += cache[mincacheid].r;
      h.g += cache[mincacheid].g;
      h.b += cache[mincacheid].b;
      h.count += cache[mincacheid].count;
      histogram[histogram_index] = h;
    }
  }
}

