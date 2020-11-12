/*! \file */
#ifndef MAIN_H_
#define MAIN_H_

#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

void ColorTransition_sRGB2linRGB(const Mat &InputMat, Mat &OutputMat);
void ColorTransition_linRGB2sRGB(const Mat &InputMat, Mat &OutputMat);
void Correction(Mat &data, const Mat &grey_point, const Mat &main_axis);
void PcaColorCorrection(const Mat &image, Mat &PCA_image);
#endif