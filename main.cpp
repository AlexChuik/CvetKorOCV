/*! \file */
#include "main.hpp"

/*!
  \brief Переводит цвета пикселей из sRGB в linRGB.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в linRGB
  \param[in] InputMat Входящее изображение с цветами пикселей в sRGB

  Принимает матрицу с форматом CV_8UC3, выдает матрицу того же размера CV_32FC3
*/
void ColorTransition_sRGB2linRGB(const Mat &InputMat, Mat &OutputMat) { 
  float lut[256];
  for (int i = 0; i < 256; i++) {
    if ((float)i/255 <= 0.04045) lut[i] = (float)i / 255 / (12.92);
    else lut[i] = pow(((float)i / 255 + 0.055) / 1.055, 2.4);
  }
  OutputMat.forEach<Vec3f>(
    [&](Vec3f &pixel, const int position[]) -> void
    {
      pixel[0] = lut[InputMat.at<Vec3b>(position[0], position[1])[0]];
      pixel[1] = lut[InputMat.at<Vec3b>(position[0], position[1])[1]];
      pixel[2] = lut[InputMat.at<Vec3b>(position[0], position[1])[2]];
    }
  );
}

/*!
  \brief Переводит цвета пикселей из linRGB в sRGB.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в sRGB 
  \param[in] InputMat Входящее изображение с цветами пикселей в linRGB

  Принимает матрицу с форматом CV_32FC3, выдает матрицу того же размера CV_32FC3
*/
void ColorTransition_linRGB2sRGB(const Mat &InputMat, Mat &OutputMat) { 
  OutputMat.forEach<Vec3f> 
  (
    [&](Vec3f &pixel, const int position[]) -> void
    {
      for (int i = 0; i < 3; i++) {
        if (InputMat.at<Vec3f>(position[0], position[1])[i] <= 0.0031308)
          pixel[i] = InputMat.at<Vec3f>(position[0], position[1])[i]* 
                      (12.92);
        else
          pixel[i] = pow(InputMat.at<Vec3f>(position[0], position[1])[i],
                      1. / 2.4) * (1.055) - 0.055;
      }                                 
    }
  );
}

/*!
  \brief Производит перобразование кластера цветов (точек в трехмерном пространстве) 
  по заданным параметрам.
  \param[in,out] data Кластер цветов, который требуется преобразовать 
  \param[in] mean Средняя точка кластера (центр масс) 
  \param[in] main_axis Главная ось кластера

  Данная функция переобразует кластер следующим образом:\n
    - вычисляет серую точку\n
    - паралельно преносит кластер так, чтобы серая точка оказалась в нуле\n
    - масштабирует оси, в результате чего главная ось становиться коллинеарна 
    галавной диагонали куба\n
    - паралельно преносит кластер на вектор (1/2, 1/2, 1/2).
*/
void Correction(Mat &data, const Scalar &mean, const Scalar &main_axis) {
  Scalar grey_point; 
  Scalar center_cube(1. / 2., 1. / 2., 1. / 2.);
  Scalar normal_vector(1. / sqrt(3), 1. / sqrt(3), 1. / sqrt(3));
  grey_point = normal_vector.dot(center_cube - mean) * main_axis;
  grey_point = mean + grey_point / normal_vector.dot(main_axis); 

  Mat scaling_mat = Mat::eye(3, 3, CV_32F);
  float sum = main_axis.dot(Scalar(1,1,1));
  scaling_mat *= sum / 3;
  scaling_mat.at<float>(0, 0) /= (main_axis[0]);
  scaling_mat.at<float>(1, 1) /= (main_axis[1]);
  scaling_mat.at<float>(2, 2) /= (main_axis[2]);

  data = data.reshape(3,0) - grey_point;
  data = data.reshape(1,0) * scaling_mat;
  data = data.reshape(3,0) + center_cube;
  data = data.reshape(1,0);
}

/*!
  \brief Производит цветокоррекцию, анализируя параметры кластера с помощью PCA. 
  \param[out] PCA_image Изображение после цветокоррекции 
  \param[in] image Входящее изображение

  Данная функция вычисляет параметры кластера цветов пикселей методом главных 
  компонент, после чего преобразует изображение посредством функции Correction.
*/
void PcaColorCorrection(const Mat &image, Mat &PCA_image) {
  int nRows = image.rows;
  int nCols = image.cols;
  ColorTransition_sRGB2linRGB(image, PCA_image);
  Mat data(nRows * nCols, 3, CV_32F);
  //формат необходимый для встроенного PCA
  data = Mat(Mat(PCA_image.t()).reshape(0, 1).t()).reshape(1, 0);
  PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 1);
  Scalar mean(Mat_<Vec3f>(pca.mean)(0,0));
  Scalar main_axis(Mat_<Vec3f>(pca.eigenvectors)(0,0));
 
  Correction(data, mean, main_axis);
  PCA_image = Mat(data.reshape(0, nCols).reshape(3, 0).t());
  ColorTransition_linRGB2sRGB(PCA_image, PCA_image);
}
int main(int argc, char **argv) {
  if (argc != 2)
  {
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }
  Mat image = imread(argv[1], IMREAD_COLOR);
  if (image.empty())
  {
    cout << "Could not open or find the image\n"
         << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  Mat PCA_image(image.size(), CV_32FC3);
  PcaColorCorrection(image, PCA_image);

  imshow("Original Image", image);
  imshow("New Image", PCA_image);
  waitKey();
  return 0;
}
