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

template< typename T >
struct TestVisualizer
{
   // sf: scaling factor, choose odd
   static void showTests(const int patch_sz, const T* tests, const int num_tests,
                         const char* caption=NULL, const char* save_to_url=NULL,
                         IplImage *img=NULL, const int sf=31, const IplImage *bg_img=NULL)
   {
      // Create image
      bool free_img = false;
      if (!img)
      {
         free_img = true;
         img = cvCreateImage(cvSize(sf*patch_sz, sf*patch_sz), IPL_DEPTH_8U, 3);
      }

      drawTests(patch_sz, tests, num_tests, img, sf, bg_img);

      // Show patch
      std::string cap = caption ? caption : utils::intToStr(num_tests) + " Tests";
      utils::showInWindow(cap.c_str(), img);

      if (save_to_url)  cvSaveImage(save_to_url, img);
      if (free_img)     cvReleaseImage(&img);
   }

   static void drawTests(const int patch_sz, const T* tests, const int num_tests,
                         IplImage *dst, const int sf=31, const IplImage *bg_img=NULL,
                         const bool zero_dst=true, const std::vector<bool>* which=NULL,
                         const CvScalar color=GRAY)
   {
      ASSURE(dst);

      const int bsh = sf/2;      // border shift: approximately centers resulting image
      //const int step = dst->widthStep;

      if (bg_img)
      {
         ASSURE_EQ(bg_img->nChannels, dst->nChannels);
         cvResize(bg_img, dst, CV_INTER_NN);

         #if 0    // 'dim' patch a little
         for (int i=0; i<dst->height; ++i)
         {
            uchar *p_ln = (uchar*)&dst->imageData[i*dst->widthStep];
            for (int j=0; j<dst->width; ++j)
            {
               for (int k=0; k<dst->nChannels; ++k, ++p_ln)
                  *p_ln = cvRound(*p_ln * 0.75);
            }
         }
         #endif
      }
      else if (zero_dst)
         memset(dst->imageData, 0, dst->height*dst->widthStep*sizeof(dst->imageData[0]));

      // Draw pixel centers as gray dots, mark central pixel
      if (sf > 1)
      {
         for (int i=0; i<patch_sz; ++i)
            for (int k=0; k<patch_sz; ++k)
               cvCircle(dst, cvPoint(i*sf+bsh, k*sf+bsh), 1, DKGRAY);
         cvCircle(dst, cvPoint(patch_sz/2*sf+bsh, patch_sz/2*sf+bsh), 5, DKGRAY);
      }

      for (int i=0; i<num_tests; ++i)
         cvLine(dst,
                cvPoint(sf*(tests[4*i] + patch_sz/2) + bsh, sf*(tests[4*i+1] + patch_sz/2) + bsh),
                cvPoint(sf*(tests[4*i+2] + patch_sz/2) + bsh, sf*(tests[4*i+3] + patch_sz/2) + bsh),
                GREEN);
   }

};
