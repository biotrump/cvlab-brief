/*
  Copyright 2010 Computer Vision Lab,
  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
  All rights reserved.

  Author: Michael Calonder (http://cvlab.epfl.ch/~calonder)
  Version: 1.0

  This file is part of the BRIEF DEMO software.

  BRIEF DEMO is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  BRIEF DEMO is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  BRIEF DEMO; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#pragma once

struct MyKeypoint
{
   int x;
   int y;
   float ori;
   float scale;
   IplImage* image;

   MyKeypoint() : x(0), y(0), image(NULL)
   { }

   MyKeypoint(int x, int y, IplImage* image) : x(x), y(y), image(image)
   { }

   MyKeypoint(int x, int y, float scale, float ori, IplImage* image)
     : x(x), y(y), ori(ori), scale(scale), image(image)
   { }

   MyKeypoint(const MyKeypoint& other)
   {
      copyFrom(other);
   }

   void operator=(const MyKeypoint& rhs)
   {
      copyFrom(rhs);
   }

   void print(const std::string prefix="", const std::string suffix="")
   {
      printf("%sMyKeypoint: (%i,%i) o=%.4f s=%.4f 0x%lX%s\n", prefix.c_str(), x, y,
             ori, scale, (unsigned long)image, suffix.c_str());
   }

private:
   // Shallow copy
   inline void copyFrom(const MyKeypoint& other)
   {
      this->x = other.x;
      this->y = other.y;
      this->ori = other.ori;
      this->scale = other.scale;
      this->image = other.image;
   }
};
