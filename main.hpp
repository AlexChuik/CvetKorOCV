/*! \file */
#ifndef MAIN_H_
#define MAIN_H_

#include <iostream>
#include <array>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "constants.hpp"

using namespace std;
using namespace cv;
using namespace constants;

void ColorTransition_sRGB2linRGB(const Mat &InputMat, Mat &OutputMat);
void ColorTransition_linRGB2sRGB(const Mat &InputMat, Mat &OutputMat);
void ColorTransition_linRGB2lab(const Mat &InputMat, Mat &OutputMat);
void ColorTransition_lab2linRGB(const Mat &InputMat, Mat &OutputMat);
void ColorTransition_lab2linRGB(const Scalar &input_pixel, 
                                Scalar &output_pixel);

void Correction(Mat &data, const Scalar &grey_point, const Scalar &main_axis);
void PcaColorCorrection(const Mat &image, Mat &PCA_image);

void MyAdd(Mat &OutputMat, const Mat &Add1, const Mat &Add2, int shift);
void HoughTransformation(Mat &Plane, int step);
void HoughAnalysis(Mat &Plane, Scalar &point_1, Scalar &point_2);
void HoughColorCorrection(const Mat &image, Mat &Hough_image);

/*!
  \brief Синглтон для таблицы подстановки.
*/
class LutSrgb2Linrgb {
 public:
  std::array<float, 256> lut;
  static LutSrgb2Linrgb& instance();
 private:
  LutSrgb2Linrgb();
  ~LutSrgb2Linrgb(); 
  LutSrgb2Linrgb(LutSrgb2Linrgb const&);
  LutSrgb2Linrgb& operator=(LutSrgb2Linrgb const&);
};
#endif
