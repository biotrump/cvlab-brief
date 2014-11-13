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

#include <cstdio>
#include <cstring>

#include <opencv/cv.h>
#include <brief/BRIEF.hpp>
#include <brief/BRIEFMatcher.hpp>
#include <brief/MyKeypoint.h>
#include <brief/GroundTruth.hpp>
#include <brief/MatchVerifier.hpp>


struct _CmdLnArgs {
   int right_img_ix;       // right image index for matching
   int desc_timing_nit;    // # iterations for timing the descriptor
   int match_timing_nit;   // # iterations for timing the matching
   int match_timing_npts;  // # pts for timing the matching

   bool parse(int argc, char *argv[])
   {
      if (argc < 3) return false;

      if (strcmp(argv[1], "--match") == 0) {
         right_img_ix = atoi(argv[2]);
         if (right_img_ix<2 || right_img_ix>6) {
            printf("[ERROR] Right image index must be in {2,...,6}\n");
            return false;
         }
      }
      else if (strcmp(argv[1], "--time-desc") == 0) {
         desc_timing_nit = atoi(argv[2]);
         if (desc_timing_nit<1 || desc_timing_nit>1000) {
            printf("[ERROR] # iterations for descriptor timing must be in {1,...,1000}\n");
            return false;
         }
      }
      else if (strcmp(argv[1], "--time-match") == 0) {
         if (argc != 4) return false;
         match_timing_nit = atoi(argv[2]);
         if (match_timing_nit<1 || match_timing_nit>1000) {
            printf("[ERROR] # iterations for descriptor timing must be in {1,...,1000}\n");
            return false;
         }
         match_timing_npts = atoi(argv[3]);
         if (match_timing_npts<1 || match_timing_npts>1000000) {
            printf("[ERROR] # points for descriptor timing must be in {1,...,1e6}\n");
            return false;
         }
      }
      else {
         printf("[ERROR] Bad command line argument: %s\n", argv[1]);
         return false;
      }

      return true;
   }

   _CmdLnArgs()
      : right_img_ix(0), desc_timing_nit(0), match_timing_nit(0), match_timing_npts(0)
   { }

} CmdLnArgs;


static void detectSURF(const IplImage* img,
                       std::back_insert_iterator< std::vector< MyKeypoint > > ins,
                       const int border)
{
   CvMemStorage* storage = cvCreateMemStorage(0);
   CvSeq* surf_kpts = NULL;

   // Using SURF's default param. Not computing descriptors
   CvSURFParams par;
   par.extended = 0;
   par.hessianThreshold = 1000;
   par.nOctaves = 3;
   par.nOctaveLayers = 4;
   cvExtractSURF(img, NULL, &surf_kpts, NULL, storage, par);

   int cnt = 0;
   for (int i=0; i < (surf_kpts?surf_kpts->total:0); i++ )
   {
      CvSURFPoint kpt = *(CvSURFPoint*)cvGetSeqElem(surf_kpts, i);

      // Reject points too close to the border, round coordinates to integers
      int x = cvRound(kpt.pt.x);
      int y = cvRound(kpt.pt.y);
      if (x<border || x>=img->width-border || y<border || y>=img->height-border)
         continue;

      MyKeypoint p(kpt.pt.x, kpt.pt.y, kpt.size, kpt.dir, const_cast< IplImage* >(img));
      *ins = p;

      ++cnt;
   }

   printf("[OK] Extracted %i keypoints\n", cnt);
}


static void drawResult(const std::vector< MyKeypoint > match_left,
                       const std::vector< MyKeypoint > match_right,
                       const IplImage *img_left, const IplImage *img_right)
{
   assert(match_left.size() == match_right.size());

   IplImage* big = utils::stackImagesVertically(img_left, img_right, true);

   for (int i=0; i<(int)match_left.size(); ++i) {
      CvPoint left = cvPoint(cvRound(match_left[i].x), cvRound(match_left[i].y));
      CvPoint right = cvPoint(cvRound(match_right[i].x), cvRound(img_left->height + match_right[i].y));
      cvCircle(big, left, 1, GREEN);
      cvCircle(big, right, 1, GREEN);
         cvLine(big, left, right, GREEN);
   }

   cvSaveImage("matches.png", big);
   printf("[OK] Matches shown in matches.png\n\n");
}


void matchImages(int right_img_ix)
{
   // The keypoint structure is expected to have members called x, y, ori,
   // scale, and image. See also MyKeypoint.h
   std::vector< MyKeypoint > left_kpts, right_kpts;

   // A descriptor is stored as a std::bitset
   BRIEF desc;
   std::vector< std::bitset< BRIEF::DESC_LEN > > feat_left, feat_right;

   // Structure providing the ground truth
   static const float TOL = 2.f;     // pixel tolerance for accepting a match
   MatchVerifier< MyKeypoint, GroundTruthMiko > verif("wall/", TOL);

   // Load images
   const IplImage *left_img = verif.getGroundTruth().getImage(1);
   const IplImage *right_img = verif.getGroundTruth().getImage(right_img_ix);
   assert(left_img);
   assert(right_img);

   // Detect, for example, SURF points
   detectSURF(left_img, std::back_inserter(left_kpts), BRIEF::INIT_PATCH_SZ);
   detectSURF(right_img, std::back_inserter(right_kpts), BRIEF::INIT_PATCH_SZ);

   // Compute descriptors
   desc.getBRIEF(left_kpts, feat_left);
   desc.getBRIEF(right_kpts, feat_right);

   // Match descriptors
   printf("[OK] Matching %i against %i descriptors...\n",
          (int)feat_left.size(), (int)feat_right.size());
   BRIEFMatcher< MyKeypoint, BRIEF::DESC_LEN > matcher;
   std::vector< MyKeypoint > match_left, match_right;
   matcher.matchLeftRight(feat_left, feat_right, left_kpts, right_kpts,
                          std::inserter(match_left, match_left.begin()),
                          std::inserter(match_right, match_right.begin()));

   // Compute percentage of correct matches
   float rr = verif.getRecognitionRate(match_left, match_right, right_img_ix);
   printf("[OK] Got %.2f%% of %i retrieved matches right\n", rr*100, (int)match_left.size());

   // Save result image
   drawResult(match_left, match_right, left_img, right_img);
}


void timeDescription(int nit)
{
   BRIEF desc;

   // Alloc nit sets of descriptors
   typedef std::vector< std::bitset< BRIEF::DESC_LEN > > DescriptorSetT;
   DescriptorSetT *many_sets = new DescriptorSetT[nit];

   // Detect points
   std::vector< MyKeypoint > some_kpts;
   IplImage *img = cvLoadImage("wall/img1.ppm", 0);
   detectSURF(img, std::back_inserter(some_kpts), BRIEF::INIT_PATCH_SZ);

   // Measure time
   double dt = utils::getMs();
   for (int i=0; i<nit; ++i)
      desc.getBRIEF(some_kpts, many_sets[i]);
   dt = utils::getMs() - dt;

   printf("[OK] Computing %i descriptors took %.3f ms (~ %.3f ms/512 desc)\n\n",
          (int)some_kpts.size(), dt/nit, 512.f/some_kpts.size() * dt/nit);

   delete [] many_sets;
   cvReleaseImage(&img);
}


void timeMatching(int nit, int npts)
{
   // Since running time is independent from the actual data, we can use
   // random binary vectors to assess matching performance
   std::bitset< BRIEF::DESC_LEN > *bits_1 = new std::bitset< BRIEF::DESC_LEN >[npts];
   std::bitset< BRIEF::DESC_LEN > *bits_2 = new std::bitset< BRIEF::DESC_LEN >[npts];
   uint16_t *matches = new uint16_t[npts];

   // Do matching
   double dt = utils::getMs();
   for (int it=0; it<nit; ++it)
   {
      for (int i=0; i<npts; ++i)
      {
         int best_cnt = BRIEF::DESC_LEN;
         int best_ix = -1;
         for (int j=0; j<npts; ++j)
         {
            int cnt = (bits_1[i] ^ bits_2[j]).count();
            if (cnt < best_cnt)
            {
               best_cnt = cnt;
               best_ix = j;
            }
         }
         matches[i] = best_ix;
      }
   }
   dt = (utils::getMs() - dt)/nit;

   printf("[OK] Matching exhaustively %i vectors (%.0e dist comp) took %.3f ms (~ %.3f ms/512 desc)\n\n",
          npts, (double)npts*npts, dt, SQR(512.f/npts)*dt);

   delete [] bits_1;
   delete [] bits_2;
   delete [] matches;
}


int main(int argc, char *argv[])
{
   if (!CmdLnArgs.parse(argc, argv))
   {
      printf("Usage: ./main (--match <ix> | --time-desc <#it> | --time-match <#it> <#pts>)\n\n");
      return 1;
   }

   if (CmdLnArgs.right_img_ix)
      matchImages(CmdLnArgs.right_img_ix);
   else if (CmdLnArgs.desc_timing_nit)
      timeDescription(CmdLnArgs.desc_timing_nit);
   else if (CmdLnArgs.match_timing_nit)
      timeMatching(CmdLnArgs.match_timing_nit, CmdLnArgs.match_timing_npts);

   return 0;
}
