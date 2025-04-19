__kernel void addToHistogramDirect(
    __global PyroXY* zin,
    __global PyroRGB*  colour,
    cfloat_t minxy, // the minimum (x,y) point captured by the histogram.
    cfloat_t maxxy, // the max (x,y) point captured by the histogram.
    __global PyroHistogramRGB* histogram,
    int image_width,
    float pixel_per_unit)
{
  int i = get_global_id(0); // The x-coordinate that we are writing to the histogram.
  
  float x = zin[i].x;
  float y = -zin[i].y; // mirror y

  if (x >= minxy.x && x < maxxy.x && y >= minxy.y && y < maxxy.y)
  {
    int pixel_x = floor((x - minxy.x) * pixel_per_unit);
    int pixel_y = floor((y - minxy.y) * pixel_per_unit);

    int indx = pixel_y * image_width + pixel_x;

    pyro_atomic_add_float(&histogram[indx].r, colour[i].r);
    pyro_atomic_add_float(&histogram[indx].g, colour[i].g);
    pyro_atomic_add_float(&histogram[indx].b, colour[i].b);
    pyro_atomic_add_float(&histogram[indx].count, 1);
  }
}
