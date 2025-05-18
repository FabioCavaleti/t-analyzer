#ifndef BOUNDING_BOX_TYPES_HPP
#define BOUNDING_BOX_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace types
{
   struct BoundingBox
   {
        int x;
        int y;
        int width;
        int height;

        BoundingBox() : x(0), y(0), width(0), height(0) {}
        BoundingBox(int x_, int y_, int width_, int height_): x(x_), y(y_), width(width_), height(height_) {}
   };


   struct Detection
   {
        BoundingBox box;
        float conf{};
        int classId{};
        std::string label;
   };

}

#endif