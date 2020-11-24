/*! \file */
#include "main.hpp"

LutSrgb2Linrgb::LutSrgb2Linrgb() {
  for (int i = 0; i < 256; i++) {
    if ((float)i/255 <= 0.04045) lut[i] = (float)i / 255 / (12.92);
    else lut[i] = pow(((float)i / 255 + 0.055) / 1.055, 2.4);
  }
}
LutSrgb2Linrgb& LutSrgb2Linrgb::instance() {
  static LutSrgb2Linrgb instance;
  return instance;
}

/*!
  \brief Переводит цвета пикселей из sRGB в linRGB.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в linRGB
  \param[in] InputMat Входящее изображение с цветами пикселей в sRGB

  Принимает матрицу с форматом CV_8UC3, выдает матрицу того же размера CV_32FC3.\n
  (B,G,R) -> (b,g,r)
*/
void ColorTransition_sRGB2linRGB(const Mat &InputMat, Mat &OutputMat) { 
  Mat T = InputMat.reshape(1,0);
  OutputMat.reshape(1,0).forEach<float>(
    [&](float &pixel, const int position[]) -> void {
        pixel = LutSrgb2Linrgb::instance().lut
                [T.at<uchar>(position[0], position[1])];                           
    }
  );
}

/*!
  \brief Переводит цвета пикселей из linRGB в sRGB.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в sRGB 
  \param[in] InputMat Входящее изображение с цветами пикселей в linRGB

  Принимает матрицу с форматом CV_32FC3, выдает матрицу того же размера CV_32FC3.\n
  (b,g,r) -> (B,G,R)
*/
void ColorTransition_linRGB2sRGB(const Mat &InputMat, Mat &OutputMat) { 
  Mat T = InputMat.reshape(1,0);
  OutputMat.reshape(1,0).forEach<float>(
    [&](float &pixel, const int position[]) -> void {
      if (T.at<float>(position[0], position[1]) <= 0.0031308)
        pixel = T.at<float>(position[0], position[1]) * 
                (12.92);
      else
        pixel = pow(T.at<float>(position[0], position[1]),
                1. / 2.4) * (1.055) - 0.055;                                
    }
  );
}

/*!
  \brief Переводит цвета пикселей из linRGB в lab.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в lab
  \param[in] InputMat Входящее изображение с цветами пикселей в linRGB

  Принимает матрицу с форматом CV_32FC3, выдает матрицу того же размера CV_32FC3.\n
  (учитывает что порядок обратный (b,g,r), выдает в правильном (l,a,b))
*/
void ColorTransition_linRGB2lab(const Mat &InputMat, Mat &OutputMat) {
  Mat temp(InputMat.size(), CV_32FC3);
  temp = InputMat.clone();
  OutputMat.forEach<Vec3f>(
    [&](Vec3f &pixel, const int position[]) -> void {
      pixel[1] = (temp.at<Vec3f>(position[0], position[1])[2] - 
                  temp.at<Vec3f>(position[0], position[1])[1]) / sqrt(2); 
      pixel[2] = (2 * temp.at<Vec3f>(position[0], position[1])[0] - 
                  temp.at<Vec3f>(position[0], position[1])[2] - 
                  temp.at<Vec3f>(position[0], position[1])[1]) / sqrt(6);                              
      pixel[0] = (temp.at<Vec3f>(position[0], position[1])[2] + 
                  temp.at<Vec3f>(position[0], position[1])[1] + 
                  temp.at<Vec3f>(position[0], position[1])[0]) / sqrt(3);   
    }
  );
}

/*!
  \brief Переводит цвета пикселей из linRGB в lab.
  \param[out] OutputMat Выходящее изображение  с цветами пикселей в linRGB
  \param[in] InputMat Входящее изображение с цветами пикселей в lab

  Принимает матрицу с форматом CV_32FC3, выдает матрицу того же размера CV_32FC3.\n
  (l,a,b) -> (b,g,r)
*/
void ColorTransition_lab2linRGB(const Mat &InputMat, Mat &OutputMat) {
  Mat temp(InputMat.size(), CV_32FC3);
  temp = InputMat.clone();
  OutputMat.forEach<Vec3f>(
    [&](Vec3f &pixel, const int position[]) -> void {
      pixel[0] = (sqrt(2) * temp.at<Vec3f>(position[0], position[1])[0] + 
                  2 * temp.at<Vec3f>(position[0], position[1])[2]) / sqrt(6); 
      pixel[1] = (sqrt(2) * temp.at<Vec3f>(position[0], position[1])[0] -
                  sqrt(3) * temp.at<Vec3f>(position[0], position[1])[1] - 
                  temp.at<Vec3f>(position[0], position[1])[2]) / sqrt(6);                              
      pixel[2] = (sqrt(2) * temp.at<Vec3f>(position[0], position[1])[0] + 
                  sqrt(3) * temp.at<Vec3f>(position[0], position[1])[1] - 
                  temp.at<Vec3f>(position[0], position[1])[2]) / sqrt(6);                                                           
    }
  );
}

/*!
  \brief Перегрузка для Scalar.
*/
void ColorTransition_lab2linRGB(const Scalar &input_pixel, 
                                Scalar &output_pixel) {
  Scalar temp = input_pixel;
  output_pixel[0] = (sqrt(2) * temp[0] + 2 * temp[2]) / sqrt(6); 
  output_pixel[1] = (sqrt(2) * temp[0] - sqrt(3) * temp[1] - 
                     temp[2]) / sqrt(6);                              
  output_pixel[2] = (sqrt(2) * temp[0] + sqrt(3) * temp[1] - 
                     temp[2]) / sqrt(6);                                                           
}

/*!
  \brief Производит перобразование кластера цветов (точек в трехмерном пространстве) 
  по заданным параметрам.
  \param[in,out] data Кластер цветов в столбец, который требуется преобразовать 
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
  Scalar grey_point(0); 
  Scalar center_cube(1. / 2., 1. / 2., 1. / 2.);
  Scalar normal_vector(1. / sqrt(3), 1. / sqrt(3), 1. / sqrt(3));
  grey_point = normal_vector.dot(center_cube - mean) * main_axis;
  grey_point = mean + grey_point / normal_vector.dot(main_axis); 

  float sum = main_axis.dot(Scalar(1,1,1));
  data = data.reshape(3,0) - grey_point;
  multiply(data, Scalar(1,1,1).div(main_axis), data, sum / 3);
  data = data + center_cube;
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
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }
  Mat image = imread(argv[1], IMREAD_COLOR);
  if (image.empty()) {
    cout << "Could not open or find the image\n"
         << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  Mat PCA_image(image.size(), CV_32FC3);
  Mat Hough_image(image.size(), CV_32FC3);
  PcaColorCorrection(image, PCA_image);
  HoughColorCorrection(image, Hough_image);

  imshow("Original Image", image);
  imshow("PCA correction", PCA_image);
  imshow("Hough correction", Hough_image);
  waitKey();
  return 0;
}
