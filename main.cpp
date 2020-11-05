#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv; 
int main( int argc, char** argv )
{
    if(argc != 2) {
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;//делаю матрицу для отображения скорректированной фотографии
    }
    Mat image = imread( argv[1], IMREAD_COLOR );
    if( image.empty() )
    {
      cout << "Could not open or find the image\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    } //получено изображение
    //теперь мы предпологаем что изображение было в sRGB и мы его хотим переделать
    //в linRGB, есть функция cvtColor, но я не понял существует ли подходящий для наших целей код
    //поэтому делаю руками (хотел с помощью LUT, но нужны матрицы CV_8U)
    
    Mat PCA_image ( image.size(), CV_32FC3 );
    image.convertTo(image,CV_32FC3,1./255);
    int nRows = image.rows;
    int nCols = image.cols;
    float *p1, *p2;
    for(int i = 0; i < nRows; ++i)
    {
        p1 = PCA_image.ptr<float>(i);
        p2 = image.ptr<float>(i);
        for (int j = 0; j < nCols*3; ++j)
        {
            if(p2[j] <= 0.04045) p1[j] = p2[j]/(12.92);
            else p1[j] = pow((p2[j]+0.055)/(1.055),2.4);
        }
    }
    //перевели в linRGB, теперь воспользуемся встроенным PCA
    Mat data(nRows*nCols,3,CV_32F);
    for( int y = 0; y < nRows; y++ ) {   //делаю вместо изначальной матрицы один столбик по 3 координаты
        for( int x = 0; x < nCols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                data.at<float>(x*nRows+y, c) = PCA_image.at<Vec3f>(y,x)[c];
            }
        }
    }
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 1);
    Mat g(1,3,CV_32F), m(1,3,CV_32F), c(1,3,CV_32F), d(1,3,CV_32F), S(3,3,CV_32F), u(1,3,CV_32F);

    pca.eigenvectors.copyTo(u); //главная ось
    pca.mean.copyTo(m); //центр масс

    for(int i=0;i<3;i++) {  //задаю параметры для адаптации 
        for(int j=0;j<3;j++) {
            S.at<float>(i,j)=0;
            if(i==j) S.at<float>(i,j) = (u.at<float>(0,0)+u.at<float>(0,1)+u.at<float>(0,2))
                                        /(3*u.at<float>(0,i));
        }
    }
    d.at<float>(0,0) = 1./sqrt(3); d.at<float>(0,1) = 1./sqrt(3); d.at<float>(0,2) = 1./sqrt(3); //ось яркости
    c.at<float>(0,0) = 1./2.; c.at<float>(0,1) = 1./2.; c.at<float>(0,2) = 1./2.; //центр куба
    
    //вычисление серой точки прсто по формуле
    Mat(m + u * Mat(( (d * ((c + (m*(-1)) ).t()))*(1./ (Mat((d*(u.t()))).at<float>(0,0)) ))).at<float>(0,0)).copyTo(g);
    
    /*cout<<pca.mean<<endl;
    cout<<pca.eigenvalues<<endl;
    cout<<pca.eigenvectors<<endl;
    cout<< g<<endl;
    cout<<S<<endl;*/

    //коррекция цвета
    for( int y = 0; y < data.rows; y++ ) {  //может есть операция для сдвига всех точек на один вектор?
        for( int k = 0; k < 3; k++ ) {
            data.at<float>(y, k) += (-1)*g.at<float>(0, k);
        }
    }
    data = Mat(data*S);
    for( int y = 0; y < data.rows; y++ ) {  
        for( int k = 0; k < 3; k++ ) {
            data.at<float>(y, k) += c.at<float>(0, k);
        }
    }


    for( int y = 0; y < nRows; y++ ) {   //делаю матрицу для отображения скорректированной фотографии
        for( int x = 0; x < nCols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                PCA_image.at<Vec3f>(y,x)[c] = data.at<float>(x*nRows+y, c) ;
            }
        }
    }

    //вернемся назад в sRGB
    for( int y = 0; y < nRows; y++ ) {  //надо ли контролировать попадание в диапозон [0,1]?
        for( int x = 0; x < nCols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                if(PCA_image.at<Vec3f>(y,x)[c] <= 0.0031308) 
                PCA_image.at<Vec3f>(y,x)[c] = PCA_image.at<Vec3f>(y,x)[c] * (12.92);
                else 
                PCA_image.at<Vec3f>(y,x)[c] = pow(PCA_image.at<Vec3f>(y,x)[c],1./2.4) * (1.055) - 0.055;
            }
        }
    }

    imshow("Original Image", image);
    imshow("New Image", PCA_image);
    waitKey();
    return 0;
}
