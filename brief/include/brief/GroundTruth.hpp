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

template< typename KeypointT >
class GroundTruthMiko
{
public:
   const int MIN_IX, MAX_IX;

   GroundTruthMiko() : MIN_IX(1), MAX_IX(6)
   { }

   // Assumes standard naming conventions for image and homography files
   GroundTruthMiko(const std::string& path) : MIN_IX(1), MAX_IX(6)
   {
      GroundTruthMiko();
      read(path);
   }

   ~GroundTruthMiko()
   {
      std::map<int, double*>::iterator it1;
      for (it1 = homographies_.begin(); it1!=homographies_.end(); ++it1)
         if (it1->second) delete [] it1->second;
   }

   // Reads all homographies and imaegs from disk
   void read(const std::string& path)
   {
      const std::string img_ext
              = (path.find("graffiti") != std::string::npos ||
                 path.find("boat") != std::string::npos ||
                 path.find("bark") != std::string::npos ||
                 path.find("trees") != std::string::npos) ? ".pgm" : ".ppm";

      for (int i=MIN_IX; i<=MAX_IX; ++i)
      {
         if (i == 1)    // usually H1to1p doesn't exist (trivially identity)
         {
            double *H = new double[9];
            memset(H, 0, 9*sizeof(double));
            H[0] = H[4] = H[8] = 1.;
            homographies_[i] = H;
         }
         else // usual case
         {
            // Read homographies
            std::string url = path + "/H1to" + utils::numToStr(i) + "p";
            ASSURE_FEX(url);

            double *H = NULL;
            utils::readMat(url, H, 3, 3);
            homographies_[i] = H;
         }

         // Read images
         img_urls_[i] = path + "/img" + utils::numToStr(i) + img_ext;
         ASSURE_FEX(img_urls_[i]);
         IplImage *img = cvLoadImage(img_urls_[i].c_str(), 0);
         assert(img);
         images_[i] = img;
      }
   }

   // Returns the corresponding image point in the right image (with index ix)
   void transform(const KeypointT& left, KeypointT& right, int ix)
   {
      assert(ix > MIN_IX && ix <= MAX_IX);
      assert((int)homographies_.size() == MAX_IX-MIN_IX+1);

      // Project point from left to right image
      double x[3] = {left.x, left.y, 1};
      double y[3];

      memset(y, 0, 3*sizeof(y[0]));

      // 3x3 matrix-vector multiply
      double *H = homographies_[ix];
      y[0] = H[0]*x[0] + H[1]*x[1] + H[2]*x[2];
      y[1] = H[3]*x[0] + H[4]*x[1] + H[5]*x[2];
      y[2] = H[6]*x[0] + H[7]*x[1] + H[8]*x[2];
      right.x = cvRound(y[0]/y[2]);
      right.y = cvRound(y[1]/y[2]);
   }

   inline IplImage* getImage(const int ix)
   {
      assert(images_.find(ix) != images_.end());
      return images_[ix];
   }

   inline const IplImage* getImage(const int ix) const
   {
      assert(images_.find(ix) != images_.end());
      return const_cast< const IplImage* >(images_.at(ix));
   }

   inline std::string getImageURL(const int ix)
   {
      assert(img_urls_.find(ix) != img_urls_.end());
      return img_urls_[ix];
   }

   static std::string getSeqName(std::string gt_path)
   {
      if (gt_path[gt_path.length()-1] == '/')
         gt_path = gt_path.substr(0, gt_path.length()-1);

      int slash_ix = gt_path.rfind("/");
      slash_ix = (slash_ix == (int)std::string::npos ? -1 : slash_ix);
      return gt_path.substr(slash_ix+1);
   }

private:
   std::map< int, double* > homographies_;     // One homography for each dataset
   std::map< int, IplImage* > images_;
   std::map< int, std::string > img_urls_;
};
