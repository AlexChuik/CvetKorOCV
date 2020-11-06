#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv; 

enum CTransition 
{
    sRGB2linRGB,
    linRGB2sRGB
};

void ColorTransition (const Mat& InputMat, Mat& OutputMat, CTransition flag);
void Korrect (Mat& data, const Mat& g, const Mat& u);
void PcaCvetKor(const Mat& image, Mat& PCA_image);

void ColorTransition (const Mat& InputMat, Mat& OutputMat, CTransition flag) //CV_32FC3 матрицы
{
    if ((InputMat.size != OutputMat.size)||(InputMat.channels() != 3)||(OutputMat.channels() != 3)) 
        return;
    if (flag == sRGB2linRGB) 
    {
        for(int i = 0; i < InputMat.rows; ++i)
        {
            for (int j = 0; j < InputMat.cols; ++j)
            {
                for (int c = 0; c < 3; ++c)
                {
                    if(InputMat.at<Vec3f>(i,j)[c] <= 0.04045) 
                        OutputMat.at<Vec3f>(i,j)[c] = InputMat.at<Vec3f>(i,j)[c]/(12.92);
                    else 
                        OutputMat.at<Vec3f>(i,j)[c] = pow((InputMat.at<Vec3f>(i,j)[c]+0.055)/(1.055),2.4);
                }
            }
        }
        return;
    }

    if (flag == linRGB2sRGB) 
    {
        for(int i = 0; i < InputMat.rows; ++i)
        {
            for (int j = 0; j < InputMat.cols; ++j)
            {
                for (int c = 0; c < 3; ++c)
                {
                    if(InputMat.at<Vec3f>(i,j)[c] <= 0.0031308) 
                        OutputMat.at<Vec3f>(i,j)[c] = InputMat.at<Vec3f>(i,j)[c] * (12.92);
                    else 
                        OutputMat.at<Vec3f>(i,j)[c] = pow(InputMat.at<Vec3f>(i,j)[c],1./2.4) * (1.055) - 0.055;
                }
            }  
        }
        return;
    }
}

void Korrect (Mat& data, const Mat& g, const Mat& u)
{
    float c_arr[] = {1./2.,1./2.,1./2.};
    Mat c(1,3,CV_32F,c_arr), S = Mat::zeros(3,3,CV_32F);

    for(int i=0;i<3;i++)  //задаю параметры для адаптации 
        S.at<float>(i,i) = (u.at<float>(0,0)+u.at<float>(0,1)+u.at<float>(0,2))/(3*u.at<float>(0,i));

    for( int y = 0; y < data.rows; y++ ) {  
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
    return;
}

void PcaCvetKor(const Mat& image, Mat& PCA_image) //принимаются и выдаются в sRGB 
{                                                 //CV_32FC3    
    int nRows = image.rows;
    int nCols = image.cols;

    ColorTransition(image, PCA_image, sRGB2linRGB);

    //перевели в linRGB, теперь воспользуемся встроенным PCA
    Mat data(nRows*nCols, 3, CV_32F);
    //делаю вместо изначальной матрицы один столбик по 3 координаты
    data = Mat(Mat(PCA_image.t()).reshape(0,1).t()).reshape(1,0);

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 1);

    Mat g(1,3,CV_32F), m(1,3,CV_32F), u(1,3,CV_32F);
    float d_arr[] = {1./sqrt(3),1./sqrt(3),1./sqrt(3)}, c_arr[] = {1./2.,1./2.,1./2.};
    Mat  c(1,3,CV_32F,c_arr), d(1,3,CV_32F,d_arr);
    pca.eigenvectors.copyTo(u); //главная ось
    pca.mean.copyTo(m); //центр масc  
    //вычисление серой точки прсто по формуле
    Mat(m + u * Mat(( (d * ((c + (m*(-1)) ).t()))*(1./ (Mat((d*(u.t()))).at<float>(0,0)) ))).at<float>(0,0)).copyTo(g);

    //коррекция цвета
    Korrect(data, g, u);
    PCA_image = Mat(data.reshape(0,nCols).reshape(3,0).t());
    //вернемся назад в sRGB
    ColorTransition(PCA_image, PCA_image, linRGB2sRGB);

}

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
    
    Mat PCA_image ( image.size(), CV_32FC3 );
    
    image.convertTo(image,CV_32FC3,1./255);
    PcaCvetKor(image, PCA_image);

    imshow("Original Image", image);
    imshow("New Image", PCA_image);
    waitKey();
    return 0;
}
