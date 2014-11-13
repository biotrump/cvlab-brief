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

#include <vector>
#include <bitset>
#include <limits>
#include <iostream>

#include <cv.h>

#include "utils.hpp"
#include "TestSampler.hpp"
#include "TestVisualizer.hpp"

#if defined(__GNUG__)      // Linux & Mac OSX
   #if __GNUC__<4 || (__GNUC__>=4 && __GNUC_MINOR__<3)   // need GCC >= 4.3.0
      #error Need at least GCC 4.3 for SSE4.2 support
   #endif
#elif defined(WIN32) || defined(_WIN32)   // Windows
   #if _MFC_VER < 0x0900   // assuming Visual Studio
      #error Need at least CC 2008 for SSE4.2 support
   #endif
#endif

struct BRIEF
{
   typedef signed char TestLocT;

   static const int DESC_LEN = 256;       // # tests, 256 should be fine
   static const int INIT_PATCH_SZ = 48;   // Size of the area surrounding feature point
   static const int INIT_KERNEL_SZ = 9;   // Box kernel size. Must be odd positive integer

   #define USE_INTEGRAL_IMAGE       1     // Use II for fast smoothing, effective for not too large images
   #if USE_INTEGRAL_IMAGE
      #define SMOOTH_ENTIRE_PATCH   0     // Smooth entire patch or only around test locations
   #endif

   // Correction for scale and orientation
   //
   // CAREFUL with this functionality. Scaling and rotating tests can cause them
   // to fall outside the image area which is only handled if DEBUG is defined.
   //
   // Also, you will have to modify interpretOri() and interpretScale() the be
   // compatible with your source of scale and orientation--an exemplary conversion
   // for OpenCV's SURF method is provided and can be enabled via SOS_SURF_OPENCV.
   #define SOS_DO_NOT_INTERPRET  0
   #define SOS_SURF_OPENCV       1
   #define SOS_ORI               SOS_DO_NOT_INTERPRET
   #define SOS_SCALE             SOS_DO_NOT_INTERPRET

   BRIEF();
   ~BRIEF();

   // Returns a vector of BRIEF descriptors, one for each keypoint in kpts
   // Assumes that all points in kpts are from the same image
   // Can provide integral image; requiring int_img->depth == IPL_DEPTH_64F
   template< typename KPT_T >
   void getBRIEF(const std::vector< KPT_T >& kpts,
                 std::vector< std::bitset<DESC_LEN> >& desc,
                 const IplImage* int_img = NULL);

   // Rotates the tests_ by phi [rad].  If `absolute`, `phi` is w.r.t.
   // the horizontal axis and otherwise w.r.t. the current orientation
   void rotateTests(const float phi, const bool absolute=true);

   // Scales all DESC_LEN tests_ in `off` by `scl`.  If `absolute`, `scl` is w.r.t.
   // the default scale (implicitely defined by DEF_KERNEL_SZ) and otherwise
   // w.r.t. the current scale
   void scaleTests(const float scl, const bool absolute=true);

   // Combined version of the above two, avoiding rounding artifacts that may
   // occur when applied sequentially
   void rotateAndScaleTests(const float phi, const float scl,
                            const bool absolute=true);

   // Apply tests and store result in bits. Use of 1st version deprecated, use 2nd!
   void applyTests(const uchar *patch_data, std::bitset<DESC_LEN>& bits,
                   const TestLocT* off) const;
   void applyTests(const float *patch_data, std::bitset<DESC_LEN>& bits,
                   const TestLocT* off) const;

   // -------------------------- experimental stuff ----------------------------

   // Hard-coded mapping from the orientation/scale a keypoint detector like
   // SURF returns to the actual rotation/scaling applied to the tests
   inline float interpretScale(const float scale) const;
   inline float interpretOri(const float ori) const;

   // Utility functions
   void printTests() const;
   void writeTests(const std::string& url) const;
   void readTests(const std::string& url);
   void getTestBounds(int bds[4]) const;
   template< typename KPT_T > bool checkAllTestInsideImage(const KPT_T& kp) const;
   template< typename KPT_T > bool checkSmoothAreaInsideIntImage(const KPT_T& kp) const;
   template< typename KPT_T > bool checkSmoothAreaForTestInsideIntImage(const KPT_T& kp) const;
   template< typename KPT_T > bool checkPatchInsideImage(const KPT_T& kp) const;

private:
   // Updates act_patch_sz_
   void recomputeActPatchSz();

   TestLocT *tests_;          // DESC_LENx4 matrix; [x1 y1 x2 y2] w.r.t. patch center
   TestLocT *tests_up_;       // dito, but upright (i.e. unrotated and unscaled)

   int act_patch_sz_;         // size of symmetric bounding box around tests, i.e. the actual patch size
   #if SOS_SCALE != SOS_DO_NOT_INTERPRET
      int act_kernel_sz_;        // size of blur kernel
   #endif

   IplImage *patch_8u_;       // buffer for INIT_PATCH_SZ x INIT_PATCH_SZ patch data
   IplImage *patch_8u_2_;     // buffer for INIT_PATCH_SZ x INIT_PATCH_SZ patch data

   #if USE_INTEGRAL_IMAGE
      struct {
         float *data;               // actual data
         const IplImage *src;       // source image
      } cur_int_img_;               // current integral image, recomputed only when needed

      float smooth_[256*256];       // patch buffer alloc'd on stack for efficiency
   #endif
};


BRIEF::BRIEF()
   : tests_(NULL), tests_up_(NULL), act_patch_sz_(0), patch_8u_(NULL), patch_8u_2_(NULL)
{
   assert(INIT_PATCH_SZ%2 == 0);
   assert(INIT_KERNEL_SZ%2 == 1);

   tests_ = new TestLocT[DESC_LEN*4];
   tests_up_ = new TestLocT[DESC_LEN*4];
   patch_8u_   = cvCreateImage(cvSize(INIT_PATCH_SZ, INIT_PATCH_SZ), IPL_DEPTH_8U, 1);
   patch_8u_2_ = cvCreateImage(cvSize(INIT_PATCH_SZ, INIT_PATCH_SZ), IPL_DEPTH_8U, 1);

   #if SOS_SCALE != SOS_DO_NOT_INTERPRET
      act_kernel_sz_ = INIT_KERNEL_SZ;
   #endif

   #if (SOS_SCALE != SOS_DO_NOT_INTERPRET || SOS_ORI != SOS_DO_NOT_INTERPRET) \
       && (!defined(DEBUG) || !DEBUG)
      #error Scale and orientation correction is experimental. Recompile in DEBUG mode.
   #endif

   #if USE_INTEGRAL_IMAGE
      cur_int_img_.data = NULL;
      cur_int_img_.src = NULL;
   #endif

   #if 0
      // Functionality allowing to use always the same set of tests--for reproducibility
      const bool always_generate = false;
      char buf[200];
      sprintf(buf, "tests_%04i.txt", DESC_LEN);
      if (utils::fileExists(buf) && !always_generate)     // read tests
         readTests(buf);
      else {
         TestSampler< TestLocT >::sample(tests_, DESC_LEN, INIT_PATCH_SZ, 1);
         writeTests(buf);
      }

   #else
      TestSampler< TestLocT >::sample(tests_, DESC_LEN, INIT_PATCH_SZ, 1);

   #endif

   memcpy(tests_up_, tests_, 4*DESC_LEN*sizeof(tests_[0]));
   recomputeActPatchSz();     // update act_patch_sz_
   printf("[OK] Actual patch size for this test configuration is %i\n", act_patch_sz_);

   //printTests();
   //TestVisualizer< TestLocT >::showTests(INIT_PATCH_SZ, tests_, DESC_LEN, NULL, NULL, NULL, 11);
}


BRIEF::~BRIEF()
{
   if (tests_) delete [] tests_; tests_ = NULL;
   if (tests_up_) delete [] tests_up_; tests_up_ = NULL;
   if (patch_8u_) cvReleaseImage(&patch_8u_); patch_8u_ = NULL;
   if (patch_8u_2_) cvReleaseImage(&patch_8u_2_); patch_8u_2_ = NULL;
}


void BRIEF::printTests() const
{
   printf("%i test locations:\n", DESC_LEN);
   for (int i=0; i<DESC_LEN; ++i)
      printf("   %3i: %4i %4i %4i %4i\n",
             i, tests_[4*i], tests_[4*i+1], tests_[4*i+2], tests_[4*i+3]);
}


template< typename KPT_T >
void BRIEF::getBRIEF(const std::vector< KPT_T >& kpts,
                     std::vector< std::bitset<DESC_LEN> >& res,
                     const IplImage* precomp_int_img)
{
   assert(kpts[0].image->nChannels == 1);

   // Make sure all points are from the same image
   const IplImage* img = kpts[0].image;
   assert(img);
   #if DEBUG
      for (int i=1; i<(int)kpts.size(); ++i)
         assert(kpts[i].image == img);
   #endif

   // Pre-compute integral image
   #if USE_INTEGRAL_IMAGE
      if (!precomp_int_img && cur_int_img_.src != img)      // Update needed for our integral image?
      {
         if (!cur_int_img_.src)
            cur_int_img_.data = new float[img->width*img->height];
         else if (cur_int_img_.src->width*cur_int_img_.src->height < img->width*img->height)
         {
            delete [] cur_int_img_.data;
            cur_int_img_.data = new float[img->width*img->height];
         }

         cur_int_img_.src = img;
         utils::computeIntegralImage(img, cur_int_img_.data);
      }
   #endif

   // Recompute bounding box, if needed
   if (act_patch_sz_ <= 0)
      recomputeActPatchSz();

   // Compute all descriptors
   std::bitset<DESC_LEN> some_desc;
   res.reserve(kpts.size());

   for (int i=0; i<(int)kpts.size(); ++i)
   {
      const KPT_T &kp = kpts[i];

      #if DEBUG
         if (!checkAllTestInsideImage(kp)) {
            some_desc.reset();
            res.push_back(some_desc);
            continue;
         }
      #endif

      // Rotate and scale tests, if information in keypoints valid
      // Note: Mind the INIT_PATCH_SZ and BOX_SZ params or this will segfault
      #if SOS_SCALE != SOS_DO_NOT_INTERPRET && SOS_ORI != SOS_DO_NOT_INTERPRET
         rotateAndScaleTests(interpretOri(kp.ori), interpretScale(kp.scale), true);
         #define KERNEL_SZ act_kernel_sz_
      #elif SOS_SCALE != SOS_DO_NOT_INTERPRET
         scaleTests(interpretScale(kp.scale), true);
         #define KERNEL_SZ act_kernel_sz_
      #elif SOS_ORI != SOS_DO_NOT_INTERPRET
         rotateTests(interpretOri(kp.ori), true);
         #define KERNEL_SZ INIT_KERNEL_SZ
      #else
         #define KERNEL_SZ INIT_KERNEL_SZ
      #endif

      // Smooth patch, compute descriptor
      #if USE_INTEGRAL_IMAGE
         assert(!precomp_int_img || precomp_int_img->depth == IPL_DEPTH_64F);

         const float* ii_data = cur_int_img_.data;
         const int ii_step = cur_int_img_.src->width;

         #if SMOOTH_ENTIRE_PATCH    // smooth entire patch, then apply tests

            #if DEBUG
               if (!checkSmoothAreaInsideIntImage(kp)) {
                  some_desc.reset();
                  res.push_back(some_desc);
                  continue;
               }
            #endif

            // Compute UNNORMALIZED smoothed patch around (kp.x, kp.y), ok since only
            // relative values are of importance
            // Using "area = B + D - A - C" where arrangement is  B  A
            //                                                    C  D
            float *p_smooth = smooth_;
            for (int v=0; v<act_patch_sz_; ++v)
            {
               // cix: smoothing window central pixel offset
               int cix = (kp.y-act_patch_sz_/2+v)*ii_step + kp.x-act_patch_sz_/2;
               const float *p_a = &ii_data[cix - KERNEL_SZ/2*ii_step + KERNEL_SZ/2];
               const float *p_b = &ii_data[cix - KERNEL_SZ/2*ii_step - KERNEL_SZ/2];
               const float *p_c = &ii_data[cix + KERNEL_SZ/2*ii_step - KERNEL_SZ/2];
               const float *p_d = &ii_data[cix + KERNEL_SZ/2*ii_step + KERNEL_SZ/2];
               for (int u=0; u<act_patch_sz_; ++u, ++p_smooth, ++p_a, ++p_b, ++p_c, ++p_d)
                  *p_smooth = ((*p_b + *p_d - *p_a - *p_c));
            }

            // Ready for tests
            res.push_back(some_desc);
            applyTests(smooth_, res[res.size()-1], tests_);

         #else    // smooth around test locations only

            #if DEBUG
               if (!checkSmoothAreaForTestInsideIntImage(kp)) {
                  some_desc.reset();
                  res.push_back(some_desc);
                  continue;
               }
            #endif

            res.push_back(some_desc);
            std::bitset<DESC_LEN> &bits = res[res.size()-1];

            const TestLocT *pxy = tests_;
            for (int j=0; j<DESC_LEN; ++j, pxy+=4)
            {
               // cix: central pixel offset
               const int cix_1 = (kp.y + pxy[1])*ii_step + kp.x+pxy[0];
               const int cix_2 = (kp.y + pxy[3])*ii_step + kp.x+pxy[2];

               // Compute tests
               bits[j] = ( ii_data[cix_1 - KERNEL_SZ/2*ii_step - KERNEL_SZ/2] +
                           ii_data[cix_1 + KERNEL_SZ/2*ii_step + KERNEL_SZ/2] -
                           ii_data[cix_1 - KERNEL_SZ/2*ii_step + KERNEL_SZ/2] -
                           ii_data[cix_1 + KERNEL_SZ/2*ii_step - KERNEL_SZ/2] )
                         >
                         ( ii_data[cix_2 - KERNEL_SZ/2*ii_step - KERNEL_SZ/2] +
                           ii_data[cix_2 + KERNEL_SZ/2*ii_step + KERNEL_SZ/2] -
                           ii_data[cix_2 - KERNEL_SZ/2*ii_step + KERNEL_SZ/2] -
                           ii_data[cix_2 + KERNEL_SZ/2*ii_step - KERNEL_SZ/2] );
            }

         #endif

      #else    // not using integral image but cvSmooth instead

         #ifdef DEBUG
            if (!checkPatchInsideImage(kp)) {
               some_desc.reset();
               res.push_back(some_desc);
               continue;
            }
         #endif

         // Need to copy patch data before smoothing
         assert(img->depth == (int)IPL_DEPTH_8U || img->depth == (int)IPL_DEPTH_8S);  // or memcpy will fail
         const int step = img->widthStep;
         uchar* start = (uchar*)&img->imageData[(kp.y-act_patch_sz_/2)*step + kp.x-act_patch_sz_/2];
         for (int j=0; j<act_patch_sz_; ++j, start += step)
            memcpy(patch_8u_->imageData + j*act_patch_sz_, start, act_patch_sz_);
         cvSmooth(patch_8u_, patch_8u_2_, CV_BLUR, KERNEL_SZ, KERNEL_SZ);

         // Ready for tests
         res.push_back(some_desc);
         applyTests((uchar*)patch_8u_2_->imageData, res[res.size()-1], tests_);
      #endif

   }

   #undef KERNEL_SZ
}


inline void BRIEF::applyTests(const uchar *patch_data,
                       std::bitset<DESC_LEN>& bits,
                       const TestLocT *txy) const
{
   assert(patch_data);
   const int PS = act_patch_sz_;
   const int PS_2 = act_patch_sz_/2;

   for (int i=0; i<DESC_LEN; ++i, txy += 4)
      bits.set(i, patch_data[(PS_2 - txy[1])*PS + txy[0]+PS_2] < patch_data[(PS_2 - txy[3])*PS + txy[2]+PS_2]);
}


inline void BRIEF::applyTests(const float *patch_data,
                       std::bitset<DESC_LEN>& bits,
                       const TestLocT *txy) const
{
   assert(patch_data);
   const int PS = act_patch_sz_;
   const int PS_2 = act_patch_sz_/2;

   for (int i=0; i<DESC_LEN; ++i, txy += 4)
      bits.set(i, patch_data[(PS_2 - txy[1])*PS + txy[0]+PS_2] < patch_data[(PS_2 - txy[3])*PS + txy[2]+PS_2]);
}


// TODO: Rotate using precomputed rotation matrix for speed
void BRIEF::rotateTests(const float phi, const bool absolute)
{
   double r, off_phi;

   TestLocT *txy =     (absolute ? tests_up_ : tests_);     // source
   TestLocT *rot_txy = tests_;                              // destination

   for (int i=0; i<2*DESC_LEN; ++i)       // for each test location
   {
      r = sqrt(SQR(txy[2*i]) + SQR(txy[2*i+1]));
      off_phi = atan2(txy[2*i+1], txy[2*i]); // + M_PI      // phi \in [-M_PI,+M_PI] --> phi \in [0,2*M_PI]
      rot_txy[2*i]   = int(r*cos(off_phi + phi));       // new 'x' coord
      rot_txy[2*i+1] = int(r*sin(off_phi + phi));
   }

   recomputeActPatchSz();

   //TestVisualizer< TestLocT >::showTests(INIT_PATCH_SZ, rot_off[i], ("Rotated tests: orientation " + utils::numToStr(i)).c_str());
}


void BRIEF::scaleTests(const float scl, const bool absolute)
{
   // debug
   //TestVisualizer< TestLocT >::showTests(INIT_PATCH_SZ, tests_, DESC_LEN,
   //                          ("before scaling: scl = " + utils::numToStr(scl)).c_str(),
   //                          "/home/mic/Desktop/unscaled.png");

   TestLocT *txy =     (absolute ? tests_up_ : tests_);     // source
   TestLocT *rot_txy = tests_;                              // destination

   for (int i=0; i<4*DESC_LEN; ++i, ++txy, ++rot_txy)
      *rot_txy = cvRound(scl * (*txy));

   // Update actual patch size
   recomputeActPatchSz();

   // Rescale kernel size
   #if SOS_SCALE != SOS_DO_NOT_INTERPRET
      if (absolute) act_kernel_sz_ = ROUND_ODD(scl*INIT_KERNEL_SZ);
      else act_kernel_sz_ = ROUND_ODD(scl*act_kernel_sz_);
   #endif
}


void BRIEF::rotateAndScaleTests(const float phi, const float scl,
                                const bool absolute)
{
   double r, off_phi;

   TestLocT *txy =     (absolute ? tests_up_ : tests_);     // source
   TestLocT *rot_txy = tests_;                              // destination

   for (int i=0; i<2*DESC_LEN; ++i)       // for each test location
   {
      r = sqrt(SQR(txy[2*i]) + SQR(txy[2*i+1]));
      off_phi = atan2(txy[2*i+1], txy[2*i]); // + M_PI      // phi \in [-M_PI,+M_PI] --> phi \in [0,2*M_PI]

      // Rotate and scale
      rot_txy[2*i]   = cvRound(scl*r*cos(off_phi + phi));       // new 'x' coord
      rot_txy[2*i+1] = cvRound(scl*r*sin(off_phi + phi));
   }

   recomputeActPatchSz();
}


void BRIEF::recomputeActPatchSz()
{
   assert(tests_);

   // Get actual patch size from test set
   TestLocT min_x = std::numeric_limits< TestLocT >::max(),
            max_x = std::numeric_limits< TestLocT >::min(),
            min_y = std::numeric_limits< TestLocT >::max(),
            max_y = std::numeric_limits< TestLocT >::min();

   TestLocT *p = tests_;
   for (int i=0; i<2*DESC_LEN; ++i)
   {
      min_x = (*p < min_x)*(*p) + (!(*p < min_x))*(min_x);
      max_x = (*p > max_x)*(*p) + (!(*p > max_x))*(max_x);
      ++p;
      min_y = (*p < min_y)*(*p) + (!(*p < min_y))*(min_y);
      max_y = (*p > max_y)*(*p) + (!(*p > max_y))*(max_y);
      ++p;
   }

   assert(min_x < 0);
   assert(max_x > 0);
   assert(min_y < 0);
   assert(max_y > 0);
   act_patch_sz_ = 2*MAX(MAX(max_x, -min_x), MAX(max_y, -min_y));
}


void BRIEF::writeTests(const std::string& url) const
{
   FILE *f = fopen(url.c_str(), "w");
   ASSURE(f);

   for (int i=0; i<DESC_LEN; ++i)
      fprintf(f, "%i %i %i %i\n", tests_[4*i], tests_[4*i+1], tests_[4*i+2], tests_[4*i+3]);

   printf("[OK] Wrote %i tests to %s\n", DESC_LEN, url.c_str());
   fclose(f);
}


void BRIEF::readTests(const std::string& url)
{
   FILE *f = fopen(url.c_str(), "r");
   ASSURE(f);
   int val;

   for (int i=0; i<4*DESC_LEN; ++i)
   {
      int res = fscanf(f, "%i", &val); if (res);
      tests_[i] = (TestLocT)val;
   }

   printf("[OK] Read %i tests from %s\n", DESC_LEN, url.c_str());
   fclose(f);
}


// Scale conversion
inline float BRIEF::interpretScale(const float scl) const
{
#if SOS_SCALE == SOS_DO_NOT_INTERPRET
   return scl;

#elif (SOS_SCALE == SOS_SURF_OPENCV)
   // OpenCV SURF uses a patch size of 8/3*kp->size
   return 8./3*scl/float(INIT_PATCH_SZ);

#else
   #error "Bad SOS_SCALE value"

#endif
}


// Orientation conversion
// Note: Function returns rotation angle for tests, not for the patch
inline float BRIEF::interpretOri(const float ori) const
{
#if SOS_ORI == SOS_DO_NOT_INTERPRET
   return ori;

#elif SOS_ORI == SOS_SURF_OPENCV
   // Three steps taking an OpenCV SURF orientation to a test rotation angle:
   //    * [deg] --> [rad]
   //    * [0,2*M_PI] --> [-M_PI,M_PI]
   //    * -ori --> +ori  because we're rotating the tests, not the patch
   // This is -(ori/180.*M_PI - M_PI) = M_PI*(1. - ori/180.)
   //return float(M_PI*(1.f - ori / 180.f));
   return -(ori - M_PI);

#else
   #error "Bad SOS_ORI value"

#endif
}

// Complains if a test around the keypoint kp falls outside image area
template< typename KPT_T >
bool BRIEF::checkAllTestInsideImage(const KPT_T& kp) const
{
   const int w = kp.image->width;
   const int h = kp.image->height;

   std::vector< int > problems;
   for (int i=0; i<2*DESC_LEN; ++i)
      if (kp.x+tests_[2*i] < 0 || kp.x+tests_[2*i] > w-1 ||
          kp.y+tests_[2*i+1] < 0 || kp.y+tests_[2*i+1] > h-1)
         problems.push_back(i/2);

   if (problems.size() > 0)
   {
      printf("Problem in %s\n  %i tests outside image kp=(%i,%i), (w,h)=(%i,%i): \n",
             __FUNCTION__, (int)problems.size(), kp.x, kp.y, w, h);
      for (int i=0; i<(int)problems.size(); ++i)
      {
         TestLocT *p = &tests_[4*problems[i]];
         printf("  test %3i:  %3i %3i %3i %3i\n", i, p[0], p[1], p[2], p[3]);
      }

      return false;
   }

   return true;
}

// Complains if smoothing window around keypoint kp drops out of the
// image area. Assuming width/height of integral image and original image
// are identical
template< typename KPT_T >
bool BRIEF::checkSmoothAreaForTestInsideIntImage(const KPT_T& kp) const
{
   int bds[4];    // xmin, ymin, xmax, ymax
   getTestBounds(bds);

   #if SOS_SCALE == SOS_DO_NOT_INTERPRET
      const int act_kernel_sz_ = INIT_KERNEL_SZ;
   #endif

   // Tricky issue with bds on second line: need to negate ymin and ymax
   if (kp.x+bds[0]-act_kernel_sz_/2-1 <= 0 || kp.x+bds[2]+act_kernel_sz_/2+1 >= kp.image->width ||
       kp.y-bds[3]-act_kernel_sz_/2-1 <= 0 || kp.y-bds[1]+act_kernel_sz_/2+1 >= kp.image->height)
   {
      printf("[WARN] BRIEF: A least one test of keypoint @(%i,%i) out of bounds, "
             "not computing descriptor\n", kp.x, kp.y);
      return false;
   }

   return true;
}


// Complains if smoothing window drops out of the image area. Same assumption
// as above
template< typename KPT_T >
bool BRIEF::checkSmoothAreaInsideIntImage(const KPT_T& kp) const
{
   assert(act_patch_sz_ > 0);
   const int S = act_patch_sz_;

   #if SOS_SCALE == SOS_DO_NOT_INTERPRET
      const int act_kernel_sz_ = INIT_KERNEL_SZ;
   #endif

   if (kp.x-S/2-act_kernel_sz_/2-1 <= 0 || kp.x+S/2+act_kernel_sz_/2+1 >= kp.image->width ||
       kp.y-S/2-act_kernel_sz_/2-1 <= 0 || kp.y+S/2+act_kernel_sz_/2+1 >= kp.image->height)
   {
      printf("[WARN] BRIEF: Keypoint @(%i,%i) causes smoothing window dropping "
             "out of image bounds\n", kp.x, kp.y);
      return false;
   }

   return true;
}


// Complains if smoothing window drops out of the image area. Same assumption
// as above
template< typename KPT_T >
bool BRIEF::checkPatchInsideImage(const KPT_T& kp) const
{
   if (kp.x < act_patch_sz_/2 || kp.y < act_patch_sz_/2 ||
       kp.x > kp.image->width - act_patch_sz_/2 || kp.y > kp.image->height - act_patch_sz_/2)
   {
      printf("[WARN] BRIEF: Keypoint @(%i,%i) in border area, zeroing descriptor\n",
             kp.x, kp.y);
      return false;
   }

   return true;
}


void BRIEF::getTestBounds(int bds[4]) const
{
   // Order: x_min, y_min, x_max, y_max
   bds[0] = bds[1] = std::numeric_limits< TestLocT >::max();
   bds[2] = bds[3] = 0;

   for (int i=0; i<2*DESC_LEN; ++i)
   {
      if (tests_[2*i] < bds[0]) bds[0] = tests_[2*i];
      if (tests_[2*i] > bds[2]) bds[2] = tests_[2*i];
      if (tests_[2*i+1] < bds[1]) bds[1] = tests_[2*i+1];
      if (tests_[2*i+1] > bds[3]) bds[3] = tests_[2*i+1];
   }
}
