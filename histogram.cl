// Calculate the histogram index of all the zin values and store it in an array histogram_cache
// This array should have the same number of elements as the render buffers.
__kernel void calcHistogramIndexJFOrig(
    __global PyroXY* zin,
    __global PyroRGB*  colour,
    cfloat_t minxy, // the minimum (x,y) point captured by the histogram.
    cfloat_t maxxy, // the max (x,y) point captured by the histogram.
    float pixel_per_unit,
    int image_width,
    __global HistogramCache* histogram_cache)
{
  int i = get_global_id(0);

  float x = zin[i].x;
  float y = -zin[i].y; // mirror y

  if (x >= minxy.x && x < maxxy.x && y >= minxy.y && y < maxxy.y)
  {
    int pixel_x = floor((x - minxy.x) * pixel_per_unit);
    int pixel_y = floor((y - minxy.y) * pixel_per_unit);
    histogram_cache[i].r = colour[i].r;
    histogram_cache[i].g = colour[i].g;
    histogram_cache[i].b = colour[i].b;
    histogram_cache[i].count = 1.0;

    // For every y add the whole width to the index. This keeps up and down in the right direction and fixes
    // images that are not square
    histogram_cache[i].hist_index = pixel_y * image_width + pixel_x;
  }
  else
  {
    histogram_cache[i].hist_index = -1;
  }
}

// The global size is the width of the image.  So we can write to the histogram without atomic methods
// because two work items will never be writing to the same histogram index.  Each work item has to loop
// over the whole buffer.
__kernel void addToHistogramJFOrig(
  __global HistogramCache* histogram_cache,
  __global PyroHistogramRGB* histogram,
  int buffer_size)
{
  int image_x = get_global_id(0); // The x-coordinate that we are writing to the histogram.
  int image_width = get_global_size(0); // The width of the image being rendered.

  for (int i = 0; i < buffer_size; i++)
  {
    int indx = histogram_cache[i].hist_index;

    if (indx >= 0 && (indx % image_width) == image_x)
    {
      histogram[indx].r += histogram_cache[i].r;
      histogram[indx].g += histogram_cache[i].g;
      histogram[indx].b += histogram_cache[i].b;
      histogram[indx].count += histogram_cache[i].count;
    }
  }
}
