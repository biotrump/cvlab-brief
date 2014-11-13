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

template< typename KeypointT, template <typename KeypointT> class GroundTruthT >
struct MatchVerifier
{
   // Expects a path to ground truth data in GroundTruthT-style and a tolerance
   // value in pixels
   MatchVerifier(const std::string path_to_gt, const float tol)
      : tol_(tol)
   {
      gt_.read(path_to_gt);
   }

   // Returns if the specified points are a match
   bool isMatch(const KeypointT& left, const KeypointT& right, const int img_ix)
   {
      KeypointT r;
      gt_.transform(left, r, img_ix);

      // L2 tolerance
      //bool res = (sqrt(SQR(r.x - right.x) + SQR(r.y - right.y)) <= tol_);

      // L1 tolerance
      bool res = (SQR(r.x - right.x) <= tol_*tol_) && (SQR(r.y - right.y) <= tol_*tol_);

      //printf("isMatch:  l(%4i,%4i) r(%4i,%4i), rp(%4i,%4i): %i\n",
      //       left.x, left.y, right.x, right.y, r.x, r.y, res);
      return res;
   }

   // Computes the fraction of correct matches
   float getRecognitionRate(const std::vector< KeypointT > match_left,
                            const std::vector< KeypointT > match_right,
                            const int img_ix)
   {
      assert(match_left.size() == match_right.size());

      int ok = 0;
      for (int i=0; i<(int)match_left.size(); ++i)
         ok += (int)isMatch(match_left[i], match_right[i], img_ix);

      return float(ok)/match_left.size();
   }

   inline const GroundTruthT< KeypointT >& getGroundTruth() const
   {
      return const_cast< const GroundTruthT< KeypointT >& >(gt_);
   }


private:
   GroundTruthT< KeypointT > gt_;
   const float tol_;
};
