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

template< typename KPT_T, int DESC_LEN >
struct BRIEFMatcher
{
   // Matching with L/R check
   void matchLeftRight(const std::vector< std::bitset< DESC_LEN > > feat_left,
                       const std::vector< std::bitset< DESC_LEN > > feat_right,
                       const std::vector< KPT_T > kpts_left,
                       const std::vector< KPT_T > kpts_right,
                       std::insert_iterator< std::vector< KPT_T > > lmatch_ins,
                       std::insert_iterator< std::vector< KPT_T > > rmatch_ins)
   {
      int *dist = new int[MAX(feat_left.size(), feat_right.size())];
      int *left_nn = new int[feat_left.size()];
      int *right_nn = new int[feat_right.size()];

      // Matching with left-right check
      int min_ix;
      for (int i=0; i<(int)feat_left.size(); ++i)
      {
         for (int k=0; k<(int)feat_right.size(); ++k)
            dist[k] = (int)(feat_left[i] ^ feat_right[k]).count();
         utils::minVect(dist, (int)feat_right.size(), &min_ix);
         left_nn[i] = min_ix;
      }
      for (int i=0; i<(int)feat_right.size(); ++i)
      {
         for (int k=0; k<(int)feat_left.size(); ++k)
            dist[k] = (int)(feat_right[i] ^ feat_left[k]).count();
         utils::minVect(dist, (int)feat_left.size(), &min_ix);
         right_nn[i] = min_ix;
      }

      for (int i=0; i<(int)feat_left.size(); ++i)
         if (right_nn[left_nn[i]] == i)
         {
            *lmatch_ins = kpts_left[i];
            *rmatch_ins = kpts_right[left_nn[i]];
            ++lmatch_ins; ++rmatch_ins;
         }

      delete [] dist;
      delete [] left_nn;
      delete [] right_nn;
   }
};
