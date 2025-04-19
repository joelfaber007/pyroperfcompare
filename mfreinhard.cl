// Calculate the histogram index of all the zin values and store it in an array histogram_cache
// This array should have the same number of elements as the render buffers.
__kernel void calcHistogramIndex(
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
    histogram_cache[i].hist_index = pixel_y * image_width + pixel_x;
  }
  else {
  histogram_cache[i].hist_index = -1;
}
}

// addToHistogram has a work item:
//   * in the 0th dimension for every render buffer
//   * in the 1st dimension for every image width pixel
//
// The local work group size should be constrained to (cl.device_info.MAX_WORK_GROUP_SIZE, 1 )
// such that only one workgroup runs at a time, and they are all accessing separate histogram_cache's
// 
// Within a work group, each work item is considering all the results that map to a unique image x value,
// thus ensuring no two work items are updating the same histogram index.
__kernel void addToHistogram(
  __global HistogramCache* histogram_cache,
  __global PyroHistogramRGB* histogram,
  int image_width) 
{
  int i = get_global_id(0); // The id of the histogram_cache we are working on
  int image_x = (get_global_id(1) + i) % image_width; // Each work group work item should check for different column of histogram pixels

  if(histogram_cache[i].hist_index % image_width == image_x && histogram_cache[i].hist_index != -1)
  {
    int indx = histogram_cache[i].hist_index;
    histogram[indx].r += histogram_cache[i].r;
    histogram[indx].g += histogram_cache[i].g;
    histogram[indx].b += histogram_cache[i].b;
    histogram[indx].count += histogram_cache[i].count;
  }  
}
