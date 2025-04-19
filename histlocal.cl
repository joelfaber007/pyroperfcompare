__kernel void addToHistogramLocal(
    __global PyroXY* zin,
    __global PyroRGB*  colour,
    cfloat_t minxy, // the minimum (x,y) point captured by the histogram.
    cfloat_t maxxy, // the max (x,y) point captured by the histogram.
    __global PyroHistogramRGB* histogram,
    int image_width,
    float pixel_per_unit)
{
  int i = get_global_id(0); // The x-coordinate that we are writing to the histogram.
  int lid = get_local_id(0);

  float x = zin[i].x;
  float y = -zin[i].y; // mirror y

  __local int indices[512]; // short cut: There are no more than 512 local work items
  __local HistogramCache cache[512];

  int myindex = -1;
  int minlocal = i; // the minimux local id with the same index into histogram

  if (x >= minxy.x && x < maxxy.x && y >= minxy.y && y < maxxy.y)
  {
    int pixel_x = floor((x - minxy.x) * pixel_per_unit);
    int pixel_y = floor((y - minxy.y) * pixel_per_unit);

    myindex = pixel_y * image_width + pixel_x;
  }
  
  indices[lid] = myindex;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (myindex > 0)
  {
    for (int otherid = lid-1; otherid >= 0; --otherid)
    {
        if (indices[otherid] == myindex)
        {
          minlocal = otherid;
        }
    }

    pyro_atomic_add_float_local(&cache[minlocal].r, colour[i].r);
    pyro_atomic_add_float_local(&cache[minlocal].g, colour[i].g);
    pyro_atomic_add_float_local(&cache[minlocal].b, colour[i].b);
    pyro_atomic_add_float_local(&cache[minlocal].count, 1.0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // If this is the lowest local id perform a bulk read of the whole histogram bin, increment and write the whole bin back.
    if (i == minlocal) {
      PyroHistogramRGB h = histogram[myindex];
      h.r += cache[i].r;
      h.g += cache[i].g;
      h.b += cache[i].b;
      h.count += cache[i].count;

      histogram[myindex] = h;
    }
  }
}

