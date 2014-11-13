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

#include "utils.hpp"

template< typename T >
struct TestSampler
{
   static void sample(T* tests, int cnt, int patch_sz, int type)
   {
      switch (type)
      {
         case 0:     // uniform random, i.i.d.
            sampleUniform(tests, patch_sz, cnt);
            break;
         case 1:     // Gaussian random, i.i.d.
            sampleGaussian(tests, patch_sz, cnt);
            break;
         default:
            STDOUT_ERROR("Bad sample type");
      }
   }

private:
   static void sampleUniform(T* tests, int patch_sz, int cnt);
   static void sampleGaussian(T* tests, int patch_sz, int cnt);

};


template< typename T >
void TestSampler< T >::sampleUniform(T* tests, int patch_sz, int cnt)
{
   for (int i=0; i<4*cnt; ++i)
      tests[i] = int(utils::randUniform()*patch_sz - patch_sz/2);
}


template< typename T >
void TestSampler< T >::sampleGaussian(T* tests, int patch_sz, int cnt)
{
   double std = patch_sz/2.4;
   int trial;

   for (int i=0; i<4*cnt; ++i) {
      do {
         trial = int(utils::randNormal(std));
      } while (trial < -patch_sz/2 || trial > patch_sz/2);
      tests[i] =  trial;
   }
}
