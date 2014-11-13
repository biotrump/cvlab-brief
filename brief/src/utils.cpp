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

#include <iostream>
#include <brief/utils.hpp>

using namespace std;

namespace utils {

LocalStorage* LocalStorage::instance;

/** Repeats the string token n times.
**/
static string repStr(string token, int n)
{
   string res="";
   for (int i=0; i<n; i++)
      res += token;
   return res;
}


/** Strips leading and trailing blanks and tabs from a string and returns
  * a copy of the result. CR (ASCII 13) and LF at EOL are treated as blanks
  * and consequently removed as well.
**/
string trim(string s)
{
   if (s.empty()) return s;

   int l=-1, r=-1;
   const char *space = " ",
              *tab = "	";    // here is a tab character in between
   const int lf = 10,   // line feed
             cr = 13;   // carriage return

   for (unsigned int i=0; i<s.length(); i++)
      if (s.substr(i,1) != space && s.substr(i,1) != tab) {
         l = i; break;
      }
   for (unsigned int i=(unsigned int)s.length()-1; i>=0; i--)
      if (s.substr(i,1) != space) {
         if (s[i] == lf || s[i] == cr)
	    continue;  // strip away cr and lf characters from the right
         r = i;
	 break;
      }
   if (l>=0 && r>=0)
      return s.substr(l, r-l+1);
   else
      return string("");
}


/** Splits a string into blocks of length at max 'width' and adds trailing
  * 'indent' spaces. If 'first_indent' is false, the indent is not applied on
  * the first line, but on all others. */
static string reformatToBlock(string str, int width, int indent, bool first_indent)
{
   string res = "";

   const string ind = repStr(" ", indent);
   size_t cut_pos;
   for (int ln=0; !str.empty(); ln++) {
      if (str.length() > (size_t)width) {
         cut_pos = str.rfind(" ", width);  // yes, "width" and not "width-1"
         if (cut_pos == string::npos) cut_pos = width;
      }
      else
         cut_pos = str.length();
      if (ln>0 || (ln==0 && first_indent)) res += ind;
      res += trim(str.substr(0, cut_pos)) + "\n";
      str = (cut_pos>=str.length()-1 ? "" : trim(str.substr(cut_pos)));
   }

   return res;
}


/** Displays an error message and halts execution after user entered a char.
  * Call via corresponding macro.
**/
void stdoutError(string msg, const char *file, int line, const char *func)
{
   string msg2 = reformatToBlock(msg, 80-2, 1, true);

   // display the message
   cout << endl
        << "##################################### ERROR ####################################" << endl
        << endl << msg2 << endl
        << "###                                                                          ###" << endl
        << " Function: " << func << endl
        << " File:     " << file << ":" << line << endl
        << "################################################################################" << endl
        << "<ENTER> to quit (c: force continue) " << flush;
   cout << endl;
   if (getchar() != 'c') exit(1);
}


/** Displays a warning and suspends execution until user enters a char. Call via
  * corresponding macro.
**/
void stdoutWarning(string msg, const char *file, int line, const char *func, bool pause)
{
   string msg2 = reformatToBlock(msg, 80-2, 1, true);

   // display the message
   cout << endl
        << "------------------------------------ WARNING -----------------------------------" << endl
        << endl << msg2 << endl
        << "---                                                                          ---" << endl
        << " Function: " << func << endl
        << " Location: " << file << ":" << line << endl
        << "--------------------------------------------------------------------------------" << endl;
   cout << endl;

   if (pause) {
      cout << "<ENTER> to continue (q: quit now) " << flush;
      if (getchar() == 'q') exit(1);
      cout << endl;
   }
}

/** Displays the image in a window and retuns the code of the key that was hit, if any.
**/
char showInWindow(const string win_name, const IplImage *image, int wait, void(*mouse_cb_func)(int,int,int,int,void*), bool destroy_win)
{
   cvNamedWindow(win_name.c_str());
   cvSetMouseCallback(win_name.c_str(), mouse_cb_func, NULL);
   cvShowImage(win_name.c_str(), image);
   char c = -1;

   c = cvWaitKey(wait);
   if (c == 'q') exit(1);
   //else if (c == 's') {
   //   printf("Enter URL to write image to (nothing to continue): ");
   //   scanf("...
   //}

   if (destroy_win) cvDestroyWindow(win_name.c_str());

   return c;
}


/** Place top image on top of bottom image and return result
**/
IplImage* stackImagesVertically(const IplImage *top, const IplImage *bottom, const bool big_in_color)
{
   ASSURE_EQ(top->depth, bottom->depth);

   const int max_w = MAX(top->width, bottom->width);

   IplImage *big;
   if (big_in_color)
   {
      big = cvCreateImage(cvSize(max_w, top->height + bottom->height), top->depth, 3);
      IplImage *top_c = cvCreateImage(cvSize(top->width,top->height), top->depth, 3);
      IplImage *bottom_c = cvCreateImage(cvSize(bottom->width,bottom->height), bottom->depth, 3);
      cvConvertImage(top, top_c, CV_GRAY2RGB);
      cvConvertImage(bottom, bottom_c, CV_GRAY2RGB);
      char *p_big = big->imageData;
      for (int i=0; i<top_c->height; ++i, p_big+=big->widthStep)
         memcpy(p_big, top_c->imageData + i*top_c->widthStep,
                top_c->widthStep);
      for (int i=0; i<bottom_c->height; ++i, p_big+=big->widthStep)
         memcpy(p_big, bottom_c->imageData + i*bottom_c->widthStep,
                bottom_c->widthStep);
      cvReleaseImage(&top_c);
      cvReleaseImage(&bottom_c);
   }
   else
   {
      big = cvCreateImage(cvSize(max_w,top->height+bottom->height), top->depth, 1);
      char *p_big = big->imageData;
      for (int i=0; i<top->height; ++i, p_big+=big->widthStep)
         memcpy(p_big, top->imageData + i*top->widthStep,
                top->widthStep);
      for (int i=0; i<bottom->height; ++i, p_big+=big->widthStep)
         memcpy(p_big, bottom->imageData + i*bottom->widthStep,
                bottom->widthStep);
   }

   return big;
}

/** Checks the existence of the file that was specified.
**/
bool fileExists(const string& file)
{
   FILE* pFile = fopen(trim(file).c_str(), "r");
   if (pFile == NULL)
      return false;
   else {
      fclose(pFile);
      return true;
   }
}


double randUniform()
{
   return cvRandReal(&LocalStorage::getInst().rng_obj);
}


double randNormal(const double std, const double mu)
{
	double x1, x2, w, y1;
	static double y2;
	static bool use_last = false;

	if (use_last)		        /* use value from previous call */
	{
		y1 = y2;
		use_last = false;
	}
	else
	{
		do {
			x1 = 2.0 * cvRandReal(&LocalStorage::getInst().rng_obj) - 1.0;
			x2 = 2.0 * cvRandReal(&LocalStorage::getInst().rng_obj) - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * log( w ) ) / w );
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = true;
	}

	return mu + y1 * std;
}


} // namespace utils

