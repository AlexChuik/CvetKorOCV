/*! \file */
#include "main.hpp"

/*!
  \brief Поэлементная сумма с сдвигом(зацикленным) по колонкам.
  \param[out] OutputMat Результат сложения
  \param[in] shift Сдвиг
  \param[in] Add1,Add2 Слагаемые 
*/
void MyAdd(Mat &OutputMat, const Mat &Add1, const Mat &Add2, int shift) {
  OutputMat.forEach<float>(
    [&](float &pixel, const int position[]) -> void {
      pixel = Add1.at<float>(position[0], position[1]) + 
              Add2.at<float>(position[0], (position[1] + shift) % Add2.cols);                                                          
    }
  );
}

/*!
  \brief Быстрое преобразование Хафа плоскости.
  \param[in,out] Plane Плоскость для преобразования
  \param[in] step Шаг в рекурсии (степень 2-х)
  \todo распараллелить для подматриц
*/
void HoughTransformation(Mat &Plane, int step = 1) {
  if (step == kDiscretization) return;
  Mat temp = Plane.clone();
  for (int k = 0; k < kDiscretization; k += 2 * step) {
    Mat temp_k(temp, Rect(0, k, Plane.cols, 2 * step));
    for (int i = 0; i < step; i++) {
      Mat plane_ki1 = Plane(Rect(0, k, Plane.cols, 2 * step)).row(2 * i);
      Mat plane_ki2 = Plane(Rect(0, k, Plane.cols, 2 * step)).row(2 * i + 1);
      MyAdd(plane_ki1, temp_k.row(i), temp_k.row(i + step), i); 
      MyAdd(plane_ki2, temp_k.row(i), temp_k.row(i + step), i + 1);  
    }
  }
  step *= 2;
  HoughTransformation(Plane, step);
}

/*!
  \brief Находит ось кластера на плоскости с помощью быстрого преобразования Хафа.
  \param[out] rand_point Вычисленная точка на главной оси
  \param[out] main_axis Направляющий вектор главной оси
  \param[in] la Проекция кластера на la 
  \param[in] lb Проекция кластера на lb

  Ищет только вертикальные прямые с наклоном вправо или влево, 
  в дискретных координатах в формате двух точек (0, [0, kDiscretization]);
  (kDiscretization, [0, kDiscretization]).
*/
void HoughAnalysis(Mat &Plane, Scalar &point_1, Scalar &point_2) {
  double max(0);
  double max_copy(0);
  Point max_point(0, 0);
  Mat Plane_copy = Plane.clone();
  HoughTransformation(Plane_copy);
  minMaxLoc(Plane_copy, NULL, &max, NULL, &max_point);
  point_1 = Scalar(0, max_point.x);
  point_2 = Scalar(kDiscretization, max_point.x + max_point.y);
  
  flip(Plane, Plane_copy, 1);
  HoughTransformation(Plane_copy);
  minMaxLoc(Plane_copy, NULL, &max_copy, NULL, &max_point);;
  if (max_copy > max) {
    max = max_copy;
    point_1 = Scalar(0, kDiscretization - max_point.x);
    point_2 = Scalar(kDiscretization, 
                     kDiscretization - max_point.x - max_point.y);
  }
}

/*!
  \brief Производит цветокоррекцию, анализируя параметры кластера с помощью PCA. 
  \param[out] Hough_image Изображение после цветокоррекции 
  \param[in] image Входящее изображение

  Данная функция вычисляет параметры кластера цветов с помощью быстрых преобразований 
  Хафа, после чего преобразует изображение посредством функции Correction.
*/
void HoughColorCorrection(const Mat &image, Mat &Hough_image) {
  ColorTransition_sRGB2linRGB(image, Hough_image);
  ColorTransition_linRGB2lab(Hough_image, Hough_image);
  float sigma(3.0f);  //параметр для размытия
  Mat la = Mat::zeros(Size(kDiscretization, kDiscretization), CV_32F);
  Mat lb = Mat::zeros(Size(kDiscretization, kDiscretization), CV_32F);
  //проецирую на плоскости la и lb
  Hough_image.forEach<Vec3f>(
    [&](Vec3f &pixel, const int position[]) -> void {
      la.at<float>((int)(pixel[0] * kDiscretization / kLAxisLenght), 
                   (int)(pixel[1] * kDiscretization / kAlphaAxisLenght + 
                   kDiscretization / 2.0f)) += 0.01f; 
      lb.at<float>((int)(pixel[0] * kDiscretization / kLAxisLenght), 
                   (int)(pixel[2] * kDiscretization / kBetaAxisLenght + 
                   kDiscretization / 2.0f)) += 0.01f;   
    }
  );
  int wight_window = (int)(6 * sigma) + ((int)(6 * sigma) + 1) % 2;
  GaussianBlur(la, la, Size(wight_window, wight_window), sigma, sigma);
  GaussianBlur(lb, lb, Size(wight_window, wight_window), sigma, sigma);
  Scalar la_point_1(0,0,0), la_point_2(0,0,0);
  Scalar lb_point_1(0,0,0), lb_point_2(0,0,0);
  Scalar main_axis(1,1,1);
  Scalar general_point(0,0,0);
  HoughAnalysis(la, la_point_1, la_point_2); 
  HoughAnalysis(lb, lb_point_1, lb_point_2);

  //получение точки и оси из дискретных координат осей на проекциях
  general_point[0] = 0.0f;
  general_point[1] = la_point_1[1] * kAlphaAxisLenght / kDiscretization - 
                     kAlphaAxisLenght / 2.0f;
  general_point[2] = lb_point_1[1] * kBetaAxisLenght / kDiscretization - 
                     kBetaAxisLenght / 2.0f;
                     
  main_axis[0] = kLAxisLenght;
  main_axis[1] = la_point_2[1] * kAlphaAxisLenght / kDiscretization - 
                 kAlphaAxisLenght / 2.0f;
  main_axis[2] = lb_point_2[1] * kBetaAxisLenght / kDiscretization - 
                 kBetaAxisLenght / 2.0f;
  main_axis = main_axis - general_point;
  
  ColorTransition_lab2linRGB(general_point, general_point);
  ColorTransition_lab2linRGB(main_axis, main_axis);
  ColorTransition_lab2linRGB(Hough_image, Hough_image);
  Mat data(image.rows * image.cols, 3, CV_32F);
  data = Mat(Mat(Hough_image.t()).reshape(0, 1).t()).reshape(1, 0);
  Correction(data, general_point, main_axis);
  Hough_image = Mat(data.reshape(0, image.cols).reshape(3, 0).t());
  ColorTransition_linRGB2sRGB(Hough_image, Hough_image);
}
